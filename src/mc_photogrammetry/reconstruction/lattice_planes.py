"""Image-only plane readings and block-lattice signatures (DS-28/DS-29).

This is the first bounded package of the DS-28 plane/lattice camera bootstrap. It
turns one frame's pixels into a small, ordered set of *plane readings* — a
candidate Manhattan plane, the side of the camera it is on, and the discrete set
of block distances its measured lattice period is consistent with — and then a
per-reading *signature*: a rectified map of structural edge energy on a
one-cell-per-block grid, which the graph stage can correlate across views.

Three things are load-bearing and were not newly invented here:

* orientation and focal length come from :mod:`calibration` (yaw modulo 90
  degrees, so the quadrant stays unresolved);
* the plane window, rectification, period and phase come from :mod:`lattice`
  and :mod:`supported`;
* region candidates come from :mod:`surfaces` (axis-labelled tiles) and
  :mod:`boundary_candidates` (axis-aligned chains), both of which are
  image-only.

What is new is only the assembly and the bookkeeping the DS-28 directive asks
for:

* the full distance-alternative set ``B = {1/16, 1/8, 1/4, 1/2, 1, 2, 3, 4, 8,
  16}`` blocks per period, pruned only by the preregistered floor
  ``support >= 0.5 * max(support)`` — both halves are needed because
  ``lattice.estimate_period`` systematically suppresses the supra-block
  harmonics that a real multi-block build pattern produces;
* a bounded rectification that never allocates above
  :data:`MAX_RECTIFIED_SIDE`, unlike ``lattice.measure_plane``'s adaptive loop
  (which can request 8192);
* a label-blind ``structural_share`` — the fraction of the region's
  contrast-weighted edge energy that is *not* explained by the plane's own
  measured lattice — so a uniform repeating wall scores near zero while a wall
  carrying an opening or a material boundary scores higher;
* a ``PlaneSignature`` whose one-cell-per-block ``cells`` map is the
  correlation surface for the optional identity cue.

Nothing here reads a manifest, a reference litematic, a frozen evaluation mask,
a GUI debug value, or any annotation. The usable-pixel domain is the caller's
derived ``~create_gui_mask`` mask and nothing else; pixels outside it are
zeroed before a single line is detected, so mutating them cannot move a reading.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
from typing import Sequence

import numpy as np

from mc_photogrammetry.reconstruction import boundary_candidates, lattice, surfaces, supported, region_growth
from mc_photogrammetry.reconstruction.calibration import (
    CalibrationResult,
    axis_directions,
    detect_segments,
    enhance,
    structural_weights,
)

DISTANCE_ALTERNATIVES: tuple[float, ...] = (
    1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0, 1.0 / 2.0, 1.0,
    2.0, 3.0, 4.0, 8.0, 16.0,
)
"""Blocks per lattice period, exactly the preregistered set ``B``.

The sub-block half comes from internal texture repeats (``estimate_period``'s
``block_supports`` at multiples 2/4/8/16, plus the ``b = 1`` block reading). The
supra-block half is a real build pattern repeating over several blocks; it has
to be added explicitly because ``lattice.HARMONIC_SHARE`` deliberately prefers
the shortest comparable autocorrelation peak and so never surfaces ``k >= 2``.
An incomplete set can produce a spuriously integral wrong answer.
"""

SUPPORT_PRUNE = 0.5
"""Preregistered support floor, relative to a reading's own best support.

An alternative survives when ``support >= SUPPORT_PRUNE * max(support)``. The
``b = 1`` reading carries support 1.0 by construction, so the floor is never
degenerate and no threshold is fitted after a reference read.
"""

MAX_READINGS = 6
"""Readings a single frame may return, per the preregistered cap."""

MAX_READINGS_PER_AXIS_SIDE = 2
"""At most one primary and one secondary reading per ``(axis, side)``."""

MIN_STRUCTURAL_SHARE = 0.15
"""Below this a reading is retained but is not eligible to fill the reading cap."""

MAX_RECTIFIED_SIDE = 2048
"""Hard cap on either side of a rectified float32 buffer (:data:`~lattice.OVERSAMPLE`)."""

MAX_SIGNATURE_CELLS = 256
"""Hard cap on each side of a signature's cell grid."""

SIGNATURE_PIXELS_PER_CELL = 8
"""Rectified pixels per signature cell before the resource cap trims it."""

STRUCTURAL_FOLD_BINS = 64
"""Bins the folded lattice profile is histogrammed into for ``structural_share``."""

MINIMUM_SEGMENTS = 12
"""Fewer detected segments than this is too little evidence to propose a plane."""

READING_ABSTENTION_REASONS: tuple[str, ...] = (
    "no_plane_region",
    "unmeasurable_plane",
    "insufficient_coverage",
)
"""Typed reasons :func:`frame_plane_readings` may record instead of a reading."""


@dataclass(frozen=True, slots=True)
class DistanceAlternative:
    """One reading of the lattice period as a distance in blocks.

    ``world_units_per_period`` is a property of the rectification only:
    ``period_pixels * lattice.rectified_pixel_step(extent, samples)`` on the
    unit-distance plane. A
    period standing for ``b`` blocks then puts the plane at ``b / that`` blocks.
    """

    blocks_per_period: float
    distance_blocks: float
    support: float


@dataclass(frozen=True, slots=True)
class PlaneRectification:
    """The image context a signature needs, carried by its reading.

    A signature is built from the image *and* from the homography that rectifies
    the reading's plane, so the reading has to remember the calibration it was
    measured with: the three world-axis directions in camera coordinates, the
    focal length, the optical centre, and the plane-window centre. All four are
    image-derived; none of them is a frame identity or a supplied geometry.
    """

    axis_matrix: tuple[tuple[float, float, float], ...]
    focal: float
    principal_point: tuple[float, float]
    centre: tuple[float, float]


@dataclass(frozen=True, slots=True)
class PlaneRegionEvidence:
    """In-memory identity and exact packed mask for one proposal-derived region.

    The crop is the proposal mask intersected with the caller's usable mask.
    Its origin is a half-open ``(x0, y0, x1, y1)`` box in the source image.
    The little-endian packed bits retain holes and disconnected components;
    they are never expanded from the polygon or its bounding box.
    """

    source_region_id: str
    axis: int
    sources: tuple[str, ...]
    image_shape: tuple[int, int]
    bbox_xyxy: tuple[int, int, int, int]
    crop_shape: tuple[int, int]
    encoding: str
    bit_order: str
    packed_mask_base64: str
    mask_sha256: str
    pixel_count: int

    def to_record(self) -> dict:
        """JSON-safe deterministic record for this in-memory evidence value."""
        return {
            "source_region_id": self.source_region_id,
            "axis": self.axis,
            "sources": list(self.sources),
            "image_shape": list(self.image_shape),
            "bbox_xyxy": list(self.bbox_xyxy),
            "crop_shape": list(self.crop_shape),
            "encoding": self.encoding,
            "bit_order": self.bit_order,
            "packed_mask_base64": self.packed_mask_base64,
            "mask_sha256": self.mask_sha256,
            "pixel_count": self.pixel_count,
        }


@dataclass(frozen=True, slots=True)
class PlaneRegionFootprint:
    """Alternative-specific sampled support of an exact source-region mask.

    Counts are row-major uint8 bytes, one per signature cell. The same integer
    ``total_samples_per_cell`` applies to every cell because sampling is square
    and uniform. This records coverage only; no cell is classified occupied.
    """

    source_region_id: str
    blocks_per_period: float
    origin_uv: tuple[float, float]
    grid_shape: tuple[int, int]
    subsample_factor: int
    inside_region_sample_counts: bytes
    valid_sample_counts: bytes
    total_samples_per_cell: int

    def __post_init__(self) -> None:
        if (not isinstance(self.source_region_id, str)
                or len(self.source_region_id) != 64
                or any(character not in "0123456789abcdef"
                       for character in self.source_region_id)
                or not isinstance(self.grid_shape, tuple)
                or len(self.grid_shape) != 2
                or any(type(value) is not int for value in self.grid_shape)
                or type(self.subsample_factor) is not int
                or type(self.total_samples_per_cell) is not int
                or not isinstance(self.origin_uv, tuple)
                or len(self.origin_uv) != 2
                or any(type(value) not in (int, float) for value in self.origin_uv)
                or type(self.blocks_per_period) not in (int, float)
                or type(self.inside_region_sample_counts) is not bytes
                or type(self.valid_sample_counts) is not bytes):
            raise ValueError("region footprint fields have invalid types")
        rows, columns = self.grid_shape
        if (rows <= 0 or columns <= 0 or self.subsample_factor <= 0
                or self.total_samples_per_cell != self.subsample_factor ** 2
                or self.total_samples_per_cell > 255
                or not all(np.isfinite(value) for value in self.origin_uv)
                or not np.isfinite(self.blocks_per_period)
                or self.blocks_per_period <= 0.0):
            raise ValueError("region footprint has invalid grid/sample dimensions")
        expected = rows * columns
        if (len(self.inside_region_sample_counts) != expected
                or len(self.valid_sample_counts) != expected):
            raise ValueError("region footprint counts do not match its grid")
        if any(inside > valid or valid > self.total_samples_per_cell
               for inside, valid in zip(
                   self.inside_region_sample_counts, self.valid_sample_counts)):
            raise ValueError("region footprint sample counts are inconsistent")

    def to_record(self) -> dict:
        """JSON-safe deterministic record for this in-memory footprint."""
        return {
            "source_region_id": self.source_region_id,
            "blocks_per_period": self.blocks_per_period,
            "origin_uv": list(self.origin_uv),
            "grid_shape": list(self.grid_shape),
            "subsample_factor": self.subsample_factor,
            "inside_region_sample_counts_base64": base64.b64encode(
                self.inside_region_sample_counts).decode("ascii"),
            "valid_sample_counts_base64": base64.b64encode(
                self.valid_sample_counts).decode("ascii"),
            "total_samples_per_cell": self.total_samples_per_cell,
            "sample_encoding": "uint8-row-major-base64",
        }


@dataclass(frozen=True, slots=True)
class PeriodicEnergyPhaseEvidence:
    """Raw periodic-energy phase measured in one rectified patch.

    ``u`` is the patch column coordinate and ``v`` is the row coordinate.  A
    missing phase is explicitly unavailable; the legacy reading field keeps
    its historical zero sentinel for that case.  This evidence describes only
    the patch's periodic energy and does not encode a block residue or world
    placement.
    """

    phase_uv_pixels: tuple[float | None, float | None]
    period_pixels: float
    patch_centre_uv: tuple[float, float]
    extent: float
    samples: int
    response_magnitude_uv: tuple[float | None, float | None]
    profile_mass_uv: tuple[float | None, float | None]
    concentration_uv: tuple[float | None, float | None]
    unavailable_reason_uv: tuple[str | None, str | None]
    source_region_id: str | None = None

    def __post_init__(self) -> None:
        pair_fields = (
            self.phase_uv_pixels,
            self.response_magnitude_uv,
            self.profile_mass_uv,
            self.concentration_uv,
            self.unavailable_reason_uv,
        )
        if any(not isinstance(values, tuple) or len(values) != 2
               for values in pair_fields):
            raise ValueError("phase evidence axis fields must be pairs")
        if (type(self.period_pixels) not in (int, float)
                or not np.isfinite(self.period_pixels) or self.period_pixels <= 0.0
                or not isinstance(self.patch_centre_uv, tuple)
                or len(self.patch_centre_uv) != 2
                or any(type(value) not in (int, float) or not np.isfinite(value)
                       for value in self.patch_centre_uv)
                or type(self.extent) not in (int, float)
                or not np.isfinite(self.extent) or self.extent <= 0.0
                or type(self.samples) is not int or self.samples < 2):
            raise ValueError("phase evidence geometry is invalid")
        if (self.source_region_id is not None
                and (not isinstance(self.source_region_id, str)
                     or len(self.source_region_id) != 64
                     or any(character not in "0123456789abcdef"
                            for character in self.source_region_id))):
            raise ValueError("phase evidence source id is invalid")

        reasons = {None, "insufficient_period", "empty_profile",
                   "nonfinite_profile", "zero_resultant", "invalid_profile",
                   "invalid_period"}
        for phase, response, mass, concentration, reason in zip(
                self.phase_uv_pixels, self.response_magnitude_uv,
                self.profile_mass_uv, self.concentration_uv,
                self.unavailable_reason_uv):
            if reason not in reasons:
                raise ValueError("phase evidence has an unknown unavailable reason")
            metrics = (response, mass, concentration)
            if any(value is not None and
                   (type(value) not in (int, float) or not np.isfinite(value))
                   for value in metrics):
                raise ValueError("phase evidence metrics must be finite or unavailable")
            if response is not None and response < 0.0:
                raise ValueError("phase response magnitude cannot be negative")
            if mass is not None and mass < 0.0:
                raise ValueError("phase profile mass cannot be negative")
            if concentration is not None and not (-1e-12 <= concentration <= 1.0 + 1e-12):
                raise ValueError("phase concentration must lie in [0, 1]")
            if (response is not None and mass is not None
                    and response > mass + 1e-9 * max(1.0, mass)):
                raise ValueError("phase response cannot exceed profile mass")
            if (response is not None and mass is not None and mass > 0.0
                    and concentration is not None
                    and not np.isclose(concentration, response / mass,
                                       rtol=1e-9, atol=1e-12)):
                raise ValueError("phase concentration disagrees with response and mass")
            if reason is None:
                if (phase is None or type(phase) not in (int, float)
                        or not np.isfinite(phase) or not (0.0 <= phase < self.period_pixels)
                        or self.period_pixels < 2.0
                        or response is None or response <= 1e-12
                        or mass is None or mass <= 1e-9
                        or concentration is None):
                    raise ValueError("measured phase lacks valid response evidence")
            elif phase is not None:
                raise ValueError("unavailable phase must not carry a phase value")
            elif reason == "insufficient_period" and self.period_pixels >= 2.0:
                raise ValueError("insufficient-period reason requires period below 2 pixels")
            elif reason == "empty_profile" and (mass is None or mass > 1e-9):
                raise ValueError("empty-profile reason requires negligible profile mass")
            elif reason == "zero_resultant" and (
                    response is None or response > 1e-12 or mass is None or mass <= 1e-9):
                raise ValueError("zero-resultant reason requires positive mass and zero response")
            elif reason in {"nonfinite_profile", "invalid_profile"} and any(
                    value is not None for value in metrics):
                raise ValueError("invalid profile cannot carry finite phase metrics")


@dataclass(frozen=True, slots=True)
class PlaneReading:
    """One frame's reading of one candidate plane.

    ``alternatives`` keeps every surviving distance interpretation, ordered by
    support then by ``blocks_per_period`` descending. Ambiguity is the honest
    output: a single rectangle cannot separate a block seam from the texel grid
    inside a block, and the integer test in :mod:`plane_graph` resolves it later
    with a second view.
    """

    frame: str
    axis: int
    side: float
    alternatives: tuple[DistanceAlternative, ...]
    period_pixels: float
    phase_pixels: tuple[float, float]
    extent: float
    samples: int
    polygon: tuple[tuple[float, float], ...] | None
    coverage_cells: int
    structural_share: float
    confidence: float
    rectification: PlaneRectification
    region_evidence: PlaneRegionEvidence | None = None
    phase_evidence: PeriodicEnergyPhaseEvidence | None = None

    def __post_init__(self) -> None:
        if self.region_evidence is not None:
            # "available" is an invariant, not just a non-None pointer. Bind
            # evidence to this reading's axis and canonical polygon on creation.
            _decode_region_evidence(
                self.region_evidence, self.axis, self.polygon)
        if self.phase_evidence is not None:
            evidence = self.phase_evidence
            geometry_equal = (
                np.isclose(evidence.period_pixels, self.period_pixels,
                           rtol=0.0, atol=1e-12)
                and evidence.patch_centre_uv == self.rectification.centre
                and np.isclose(evidence.extent, self.extent, rtol=0.0, atol=1e-12)
                and evidence.samples == self.samples
            )
            source_equal = (
                evidence.source_region_id == (
                    self.region_evidence.source_region_id
                    if self.region_evidence is not None else None)
            )
            legacy_equal = all(
                np.isclose(float(legacy),
                           float(phase if phase is not None else 0.0),
                           rtol=0.0, atol=1e-12)
                for legacy, phase in zip(self.phase_pixels,
                                         evidence.phase_uv_pixels)
            )
            if not (geometry_equal and source_equal and legacy_equal):
                raise ValueError("phase evidence is not bound to this reading")

    @property
    def region_provenance_status(self) -> str:
        """Whether this reading retains exact proposal-derived region evidence."""
        return "available" if self.region_evidence is not None else "unavailable"

    def distance_for(self, blocks_per_period: float) -> float | None:
        """Return the unique retained distance for ``b``, or ``None``.

        Missing and ambiguous alternatives are both a typed refusal. In
        particular, never select the first of duplicate/manual alternatives.
        """
        matches = self.matching_alternatives(blocks_per_period)
        if len(matches) != 1:
            return None
        try:
            distance = float(matches[0].distance_blocks)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return None
        return distance if np.isfinite(distance) and distance > 0.0 else None

    def matching_alternatives(
        self, blocks_per_period: float,
    ) -> tuple[DistanceAlternative, ...]:
        """Return alternatives within V1's canonical b tolerance.

        The helper is shared by signatures and graph geometry so both use the
        same unique-match contract. Malformed alternative labels fail closed.
        """
        try:
            requested = float(blocks_per_period)
        except (TypeError, ValueError, OverflowError):
            return ()
        if not np.isfinite(requested) or requested <= 0.0:
            return ()
        matches: list[DistanceAlternative] = []
        for alternative in self.alternatives:
            try:
                candidate = float(alternative.blocks_per_period)
            except (AttributeError, TypeError, ValueError, OverflowError):
                return ()
            if not np.isfinite(candidate) or candidate <= 0.0:
                return ()
            tolerance = (1e-9 + 4.0 * np.finfo(float).eps
                         * max(1.0, abs(requested), abs(candidate)))
            if abs(requested - candidate) <= tolerance:
                matches.append(alternative)
        return tuple(matches)


@dataclass(frozen=True, slots=True)
class PlaneSignature:
    """A reading's rectified structural map on a one-cell-per-block grid.

    ``cells`` is the seam-band energy per cell, ``interior`` the cell-interior
    energy, ``coverage`` which cells actually came from usable pixels, and
    ``origin_uv`` the ``(u, v)`` block coordinate of cell ``(0, 0)``. The map is
    not evidence of plane identity on its own — a uniform repeating wall
    correlates at every integer shift — which is why the graph treats its peak as
    an optional scored cue and refuses on repetition. ``region_footprint`` is
    separate coverage evidence for the exact proposal mask on this alternative;
    it is not occupancy and does not change graph scoring.
    """

    reading: PlaneReading
    blocks_per_period: float
    cells: np.ndarray
    interior: np.ndarray
    coverage: np.ndarray
    origin_uv: tuple[float, float]
    region_footprint: PlaneRegionFootprint | None = None


@dataclass(frozen=True, slots=True)
class SignatureSamplingWitnessV1:
    """Exact evaluation-only source dependencies of one plane signature."""

    schema: str
    rgb_rectification: lattice.RectificationWitnessV1
    coverage_rectification: lattice.RectificationWitnessV1
    region_rectification: lattice.RectificationWitnessV1 | None
    grayscale_sha256: str
    gradient_y_sha256: str
    gradient_x_sha256: str
    energy_sha256: str
    coverage_patch_sha256: str
    gradient_edge_order: int
    subsample_factor: int
    seam_band: int
    cell_rectangles: tuple[tuple[int, int, int, int], ...]
    interior_rectangles: tuple[tuple[int, int, int, int], ...]
    cell_source_ptr: np.ndarray
    cell_source_indices: np.ndarray
    witness_sha256: str = ""

    def __post_init__(self) -> None:
        if (self.schema != "mc-photogrammetry-signature-sampling-witness-v1"
                or self.gradient_edge_order != 1
                or self.subsample_factor <= 0 or self.seam_band <= 0
                or self.cell_source_ptr.dtype != np.int64
                or self.cell_source_indices.dtype != np.int64
                or self.cell_source_ptr.flags.writeable
                or self.cell_source_indices.flags.writeable
                or self.cell_source_ptr[-1] != len(self.cell_source_indices)
                or len(self.cell_rectangles) + 1 != len(self.cell_source_ptr)
                or len(self.cell_rectangles) != len(self.interior_rectangles)):
            raise ValueError("signature sampling witness is malformed or mutable")
        digest = hashlib.sha256(json.dumps(
            self.to_record(include_digest=False), sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if self.witness_sha256 not in ("", digest):
            raise ValueError("signature sampling witness digest mismatch")
        object.__setattr__(self, "witness_sha256", digest)

    def to_record(self, *, include_digest: bool = True) -> dict:
        def row(value: np.ndarray) -> dict:
            return {"dtype": value.dtype.str, "shape": list(value.shape),
                    "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}
        result = {
            "schema": self.schema,
            "rgb_rectification_sha256": self.rgb_rectification.witness_sha256,
            "coverage_rectification_sha256": self.coverage_rectification.witness_sha256,
            "region_rectification_sha256": None if self.region_rectification is None
            else self.region_rectification.witness_sha256,
            "grayscale_sha256": self.grayscale_sha256,
            "gradient_y_sha256": self.gradient_y_sha256,
            "gradient_x_sha256": self.gradient_x_sha256,
            "energy_sha256": self.energy_sha256,
            "coverage_patch_sha256": self.coverage_patch_sha256,
            "gradient_edge_order": self.gradient_edge_order,
            "subsample_factor": self.subsample_factor, "seam_band": self.seam_band,
            "cell_rectangles": [list(value) for value in self.cell_rectangles],
            "interior_rectangles": [list(value) for value in self.interior_rectangles],
            "cell_source_ptr": row(self.cell_source_ptr),
            "cell_source_indices": row(self.cell_source_indices)}
        if include_digest:
            result["witness_sha256"] = self.witness_sha256
        return result


@dataclass(frozen=True, slots=True)
class ReadingAbstention:
    """A candidate region that did not become a reading, and the typed reason."""

    frame: str
    axis: int | None
    side: float | None
    reason: str
    detail: str = ""


def _canonical_region_polygon(polygon: tuple[tuple[float, float], ...] | None) -> tuple:
    if polygon is None:
        raise ValueError("proposal region evidence requires a polygon")
    points = tuple((float(point[0]), float(point[1])) for point in polygon)
    if (len(points) < 3
            or any(not np.isfinite(value) for point in points for value in point)):
        raise ValueError("proposal region polygon is invalid")
    # Treat cyclic starts and winding direction as the same polygon while
    # preserving every supplied vertex. Normalize signed zero for stable JSON.
    points = tuple(tuple(0.0 if value == 0.0 else value for value in point)
                   for point in points)
    if len(points) > 3 and points[0] == points[-1]:
        points = points[:-1]
    variants = []
    for ordered in (points, tuple(reversed(points))):
        variants.extend(ordered[index:] + ordered[:index]
                        for index in range(len(ordered)))
    return min(variants)


def _region_mask_digest(
    image_shape: tuple[int, int], bbox_xyxy: tuple[int, int, int, int],
    crop_shape: tuple[int, int], packed_mask: bytes,
) -> str:
    header = json.dumps({
        "image_shape": list(image_shape),
        "bbox_xyxy": list(bbox_xyxy),
        "crop_shape": list(crop_shape),
        "encoding": "packbits-base64",
        "bit_order": "little",
    }, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    return hashlib.sha256(header + b"\0" + packed_mask).hexdigest()


def _region_source_id(
    axis: int,
    sources: tuple[str, ...],
    image_shape: tuple[int, int],
    bbox_xyxy: tuple[int, int, int, int],
    crop_shape: tuple[int, int],
    polygon: tuple[tuple[float, float], ...],
    mask_sha256: str,
) -> str:
    identity = {
        "schema": "plane-region-evidence-v1",
        "axis": int(axis),
        "sources": list(sources),
        "image_shape": list(image_shape),
        "bbox_xyxy": list(bbox_xyxy),
        "crop_shape": list(crop_shape),
        "polygon": [[x, y] for x, y in polygon],
        "encoding": "packbits-base64",
        "bit_order": "little",
        "mask_sha256": mask_sha256,
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _make_region_evidence(
    mask: np.ndarray,
    axis: int,
    polygon: tuple[tuple[float, float], ...],
    sources: tuple[str, ...],
) -> PlaneRegionEvidence:
    """Pack an already-usable-intersected mask with a content-addressed id."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or binary.size == 0 or not binary.any():
        raise ValueError("region evidence requires a non-empty 2D mask")
    canonical_polygon = _canonical_region_polygon(polygon)
    canonical_sources = tuple(sorted(sources))
    if any(not isinstance(source, str) for source in canonical_sources):
        raise ValueError("region evidence sources must be strings")
    rows, columns = np.nonzero(binary)
    x0, x1 = int(columns.min()), int(columns.max()) + 1
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    bbox = (x0, y0, x1, y1)
    crop = np.ascontiguousarray(binary[y0:y1, x0:x1])
    crop_shape = (int(crop.shape[0]), int(crop.shape[1]))
    packed = np.packbits(crop.reshape(-1), bitorder="little").tobytes()
    image_shape = (int(binary.shape[0]), int(binary.shape[1]))
    mask_sha = _region_mask_digest(image_shape, bbox, crop_shape, packed)
    source_id = _region_source_id(
        int(axis), canonical_sources, image_shape, bbox, crop_shape,
        canonical_polygon, mask_sha)
    evidence = PlaneRegionEvidence(
        source_region_id=source_id,
        axis=int(axis),
        sources=canonical_sources,
        image_shape=image_shape,
        bbox_xyxy=bbox,
        crop_shape=crop_shape,
        encoding="packbits-base64",
        bit_order="little",
        packed_mask_base64=base64.b64encode(packed).decode("ascii"),
        mask_sha256=mask_sha,
        pixel_count=int(np.count_nonzero(binary)),
    )
    _decode_region_evidence(evidence, int(axis), canonical_polygon)
    return evidence


def _decode_region_evidence(
    evidence: PlaneRegionEvidence,
    axis: int,
    polygon: tuple[tuple[float, float], ...] | None,
) -> np.ndarray:
    """Validate and reconstruct the exact post-usable source mask."""
    if not isinstance(evidence, PlaneRegionEvidence):
        raise ValueError("reading region evidence has an invalid type")
    if (type(evidence.source_region_id) is not str
            or len(evidence.source_region_id) != 64
            or any(value not in "0123456789abcdef" for value in evidence.source_region_id)
            or type(evidence.mask_sha256) is not str
            or len(evidence.mask_sha256) != 64
            or any(value not in "0123456789abcdef" for value in evidence.mask_sha256)
            or type(evidence.packed_mask_base64) is not str
            or type(evidence.encoding) is not str
            or type(evidence.bit_order) is not str
            or not isinstance(evidence.sources, tuple)
            or any(type(value) is not str for value in evidence.sources)
            or tuple(sorted(evidence.sources)) != evidence.sources
            or not isinstance(evidence.image_shape, tuple)
            or not isinstance(evidence.crop_shape, tuple)
            or not isinstance(evidence.bbox_xyxy, tuple)):
        raise ValueError("reading region evidence fields are malformed")
    if evidence.encoding != "packbits-base64" or evidence.bit_order != "little":
        raise ValueError("reading region mask encoding is unsupported")
    if (type(evidence.axis) is not int or evidence.axis != int(axis)
            or len(evidence.image_shape) != 2 or len(evidence.crop_shape) != 2
            or len(evidence.bbox_xyxy) != 4):
        raise ValueError("reading region evidence dimensions or axis are invalid")
    height, width = evidence.image_shape
    x0, y0, x1, y1 = evidence.bbox_xyxy
    crop_height, crop_width = evidence.crop_shape
    dims = (height, width, x0, y0, x1, y1, crop_height, crop_width,
            evidence.pixel_count)
    if any(type(value) is not int for value in dims):
        raise ValueError("reading region evidence dimensions must be integers")
    if (height <= 0 or width <= 0 or x0 < 0 or y0 < 0 or x1 > width or y1 > height
            or x1 <= x0 or y1 <= y0 or (crop_height, crop_width) != (y1 - y0, x1 - x0)
            or evidence.pixel_count <= 0):
        raise ValueError("reading region evidence crop is out of bounds")
    try:
        packed = base64.b64decode(evidence.packed_mask_base64, validate=True)
    except (TypeError, ValueError) as error:
        raise ValueError("reading region mask base64 is malformed") from error
    bit_count = crop_height * crop_width
    expected_bytes = (bit_count + 7) // 8
    if len(packed) != expected_bytes:
        raise ValueError("reading region packed-mask length is invalid")
    if base64.b64encode(packed).decode("ascii") != evidence.packed_mask_base64:
        raise ValueError("reading region packed-mask base64 is not canonical")
    if bit_count % 8 and packed[-1] >> (bit_count % 8):
        raise ValueError("reading region packed-mask padding bits are nonzero")
    bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="little")
    crop = bits[:bit_count].reshape(crop_height, crop_width).astype(bool)
    if int(np.count_nonzero(crop)) != evidence.pixel_count:
        raise ValueError("reading region pixel count does not match packed mask")
    # A minimal bounding rectangle may contain entirely empty interior rows or
    # columns (for example, disconnected islands). Only the four outer edges
    # must contain a foreground pixel.
    if not (crop[0].any() and crop[-1].any() and crop[:, 0].any() and crop[:, -1].any()):
        raise ValueError("reading region crop is not the minimal nonzero bbox")
    mask_sha = _region_mask_digest(
        evidence.image_shape, evidence.bbox_xyxy, evidence.crop_shape, packed)
    if mask_sha != evidence.mask_sha256:
        raise ValueError("reading region mask digest mismatch")
    sources = evidence.sources
    canonical_polygon = _canonical_region_polygon(polygon)
    expected_id = _region_source_id(
        evidence.axis, sources, evidence.image_shape, evidence.bbox_xyxy,
        evidence.crop_shape, canonical_polygon, mask_sha)
    if expected_id != evidence.source_region_id:
        raise ValueError("reading source-region identity mismatch")
    full = np.zeros(evidence.image_shape, dtype=bool)
    full[y0:y1, x0:x1] = crop
    return full


def rectification_homography(
    axis: int,
    axis_matrix: np.ndarray,
    focal: float,
    principal_point: tuple[float, float],
    side: float,
    distance: float,
) -> np.ndarray:
    """The 3x3 map from block coordinates on a plane to image pixels.

    ``lattice.rectify_plane`` builds the same projection by hand over a pixel
    grid; this exposes it as a homography. A point ``(u, v)`` in **blocks** on
    the plane at ``distance`` blocks sits at camera point
    ``distance*side*normal + u*first + v*second``, which the shared pinhole
    projection turns into pixels. Composing two of these (and their inverses) is
    how two readings' cell grids are brought into one frame.

    No ``cv2.findHomography`` is involved: this is the closed form the existing
    rectifier already uses, and every coefficient is measured or declared.
    """
    if distance <= 0.0 or not np.isfinite(distance):
        raise ValueError("distance must be a positive finite number")
    if focal <= 0.0 or not np.isfinite(focal):
        raise ValueError("focal must be a positive finite number")
    axis_matrix = np.asarray(axis_matrix, dtype=float)
    normal, first, second = lattice.plane_basis(axis, axis_matrix)
    centre_x, centre_y = principal_point
    projection = np.asarray(
        ((focal, 0.0, centre_x), (0.0, -focal, centre_y), (0.0, 0.0, 1.0)),
        dtype=float,
    )
    basis = np.column_stack((first, second, distance * side * normal))
    return projection @ basis


def _grayscale(image: np.ndarray) -> np.ndarray:
    frame = np.asarray(image)
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] >= 3:
        import cv2

        return cv2.cvtColor(frame[..., :3], cv2.COLOR_RGB2GRAY)
    raise ValueError(f"expected a grey or RGB image, got shape {frame.shape}")


def _masked_grey(image: np.ndarray, usable: np.ndarray) -> np.ndarray:
    """Grey image with every non-usable pixel forced to a constant.

    Done before segment detection so that no line can be found on excluded
    pixels, and so that changing excluded RGB cannot change a reading. The mask's
    own boundary is a function of the mask and the usable content alone.
    """
    grey = _grayscale(image).copy()
    grey[~usable] = 0
    return grey


def _region_polygon(tiles: Sequence[tuple[int, int]], tile_size: int) -> tuple[tuple[float, float], ...]:
    rows = [row for row, _ in tiles]
    columns = [column for _, column in tiles]
    x0 = float(min(columns) * tile_size)
    y0 = float(min(rows) * tile_size)
    x1 = float((max(columns) + 1) * tile_size)
    y1 = float((max(rows) + 1) * tile_size)
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def _polygon_mask(polygon: Sequence[Sequence[float]], height: int, width: int) -> np.ndarray:
    import cv2

    mask = np.zeros((height, width), dtype=np.float32)
    points = np.asarray(polygon, dtype=np.float64)
    if points.ndim != 2 or len(points) < 3:
        return mask
    cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1.0)
    return mask


def _overlap_fraction(first: np.ndarray, second: np.ndarray) -> float:
    intersection = float(np.count_nonzero((first > 0.5) & (second > 0.5)))
    if intersection == 0:
        return 0.0
    smaller = min(float(np.count_nonzero(first > 0.5)), float(np.count_nonzero(second > 0.5)))
    return intersection / smaller if smaller > 0 else 0.0


def _candidate_regions(
    grey: np.ndarray,
    usable: np.ndarray,
    calibration_result: CalibrationResult,
    segments: np.ndarray,
    weights: np.ndarray,
    diagnostics: dict | None,
) -> list[tuple[int, np.ndarray, tuple[tuple[float, float], ...], float]]:
    """Axis-labelled region candidates from the two image-only sources, merged.

    The tile source (:func:`surfaces.planar_regions`) and the chain source
    (:func:`boundary_candidates.propose_boundary_candidates`) are merged by
    ``(axis, overlap >= 0.5)`` keeping the higher of the two scores. Chains are
    used only to *bound* a region, never to choose the plane — the M4-C lesson.
    ``grey`` is the already-masked single-channel frame, so no line can be
    detected on an excluded pixel by either source.
    """
    height, width = usable.shape
    focal = float(calibration_result.focal_pixels)
    yaw = float(calibration_result.yaw_degrees)
    pitch = float(calibration_result.pitch_degrees)
    roll = float(calibration_result.roll_degrees)
    principal = calibration_result.principal_point

    candidates: list[tuple[int, np.ndarray, tuple[tuple[float, float], ...], float]] = []
    tile_regions = surfaces.planar_regions(segments, focal, yaw, pitch, width, height)
    for region in tile_regions:
        mask = region.mask(height, width) * usable
        if float(mask.sum()) <= 0.0:
            continue
        candidates.append(
            (int(region.axis), mask, _region_polygon(region.tiles, region.tile_size),
             float(region.confidence))
        )

    chain_stats: dict = {}
    chains = boundary_candidates.find_chains(
        grey, focal=focal, yaw_degrees=yaw, pitch_degrees=pitch,
        roll_degrees=roll, principal_point=principal,
    )
    boundary = boundary_candidates.propose_boundary_candidates(
        chains, width, height, stats=chain_stats)
    strengths = [float(candidate.strength) for candidate in boundary]
    reference = max(strengths) if strengths else 0.0
    for candidate in boundary:
        normal = [axis for axis in range(3) if axis not in candidate.axis_pair]
        if len(normal) != 1:
            continue
        mask = _polygon_mask(candidate.polygon, height, width) * usable
        if float(mask.sum()) <= 0.0:
            continue
        score = float(candidate.strength) / reference if reference > 0.0 else 0.0
        candidates.append((int(normal[0]), mask, tuple(candidate.polygon), min(score, 1.0)))

    merged: list[tuple[int, np.ndarray, tuple[tuple[float, float], ...], float]] = []
    for axis, mask, polygon, score in candidates:
        replaced = False
        for index, (other_axis, other_mask, _polygon, other_score) in enumerate(merged):
            if other_axis != axis:
                continue
            if _overlap_fraction(mask, other_mask) >= 0.5:
                if score > other_score:
                    merged[index] = (axis, mask, polygon, score)
                replaced = True
                break
        if not replaced:
            merged.append((axis, mask, polygon, score))
    if diagnostics is not None:
        diagnostics.setdefault("planar_regions", len(tile_regions))
        diagnostics.setdefault("boundary_candidates", len(boundary))
        diagnostics.setdefault("chain_stats", chain_stats)
    return merged


def _bounded_measure(
    image: np.ndarray,
    axis: int,
    axis_matrix: np.ndarray,
    focal: float,
    *,
    mask: np.ndarray,
    centre: tuple[float, float],
    principal_point: tuple[float, float],
    extent: float,
    side: float,
    max_rectified_side: int,
) -> tuple["lattice.PeriodEstimate | None", int, np.ndarray | None, np.ndarray | None, int | None]:
    """``lattice.measure_plane``'s adaptive loop under a hard buffer cap.

    Mirrors the three-round logic exactly — measure, use the implied distance to
    choose a sampling that would put the finest detail at ``TARGET_PERIOD``
    pixels, measure again — but clamps every request to
    ``max_rectified_side``. The returned period, sampling and extent are all from
    the *same* round, so the distance derived from them is consistent; the two
    patches come back so the caller can compute phases and the structural share
    without rectifying a third time. The final value is the winning profile axis
    required for higher-order support.
    """
    reason = "exception"
    executed_rounds = 0
    try:
        samples = int(min(max(64, lattice.sampling_for(extent, focal)), max_rectified_side))
        found: "lattice.PeriodEstimate | None" = None
        winning_direction: int | None = None
        patch = weights_patch = None
        measured_samples = samples
        parent = (region_growth._period_scalar_state.get()
                  if region_growth._attrition_state.get() is not None else None)
        for _round in range(3):
            measured_samples = samples
            executed_rounds = _round + 1
            round_reason = "exception"
            image_status = "not_evaluated"
            mask_status = "not_evaluated"
            mask_mass = None
            round_found = round_direction = None
            round_identity = (*parent["identity"], _round) if parent is not None else None
            with region_growth._scalar_period_context(identity=round_identity, axis=axis):
                region_growth._scalar_period_expect("period_round")
                try:
                    lattice._period_support_event("measurement_round_begin", round_index=_round, samples=samples, extent=extent, centre=centre, axis=axis, side=side)
                    image_status = "attempted"
                    patch = lattice.rectify_plane(
                        image, axis, axis_matrix, focal, extent=extent, samples=samples,
                        centre=centre, principal_point=principal_point, side=side)
                    image_status = "failed" if patch is None else "succeeded"
                    if patch is None:
                        reason = round_reason = "image_rectification_failed"
                        return None, samples, None, None, None
                    mask_status = "attempted"
                    weights_patch = lattice.rectify_plane(
                        mask, axis, axis_matrix, focal, extent=extent, samples=samples,
                        centre=centre, principal_point=principal_point, side=side)
                    mask_status = "failed" if weights_patch is None else "succeeded"
                    if weights_patch is None or (mask_mass := float(weights_patch.sum())) <= 0.0:
                        reason = round_reason = ("mask_rectification_failed" if weights_patch is None else "zero_mask_support")
                        return None, samples, None, None, None
                    region_growth._scalar_period_expect("period_reconciliation")
                    period_result = lattice._plane_period_with_direction(patch, weights=weights_patch)
                    if period_result is None:
                        found, winning_direction = None, None
                    else:
                        found, winning_direction = period_result
                    round_found, round_direction = found, winning_direction
                    lattice._period_support_event("measurement_round", round_index=_round, samples=samples, extent=extent, centre=centre, axis=axis, side=side, patch=patch, region_weights=weights_patch, estimate=found, winning_direction=winning_direction)
                    if found is None:
                        reason = round_reason = "no_directional_period"
                        return None, samples, patch, weights_patch, None
                    if found.period_pixels >= lattice.TARGET_PERIOD:
                        reason = round_reason = "target_period_reached"
                        break
                    wanted = int(round(samples * lattice.TARGET_PERIOD / max(found.period_pixels, 1e-6)))
                    wanted = int(min(max(64, wanted), max_rectified_side))
                    if wanted <= samples:
                        reason = round_reason = "resampling_cap"
                        break
                    samples = wanted
                    reason = round_reason = "round_limit" if _round == 2 else "resample"
                finally:
                    if parent is not None:
                        with region_growth._scalar_observation_guard(region_growth._attrition_state.get()):
                            region_growth._scalar_period_finish("period_round", round_reason,
                                round_index=_round, samples=measured_samples, extent=float(extent),
                                image_rectification="exception" if image_status == "attempted" else image_status,
                                mask_rectification="exception" if mask_status == "attempted" else mask_status,
                                mask_mass=mask_mass, mask_mass_state="measured" if mask_mass is not None else "not_evaluated",
                                selected_period=float(round_found.period_pixels) if round_found is not None else None,
                                selected_strength=float(round_found.strength) if round_found is not None else None,
                                selected_direction=round_direction,
                                selected_period_state="measured" if round_found is not None else "not_evaluated",
                                selected_strength_state="measured" if round_found is not None else "not_evaluated",
                                selected_direction_state="measured" if round_direction is not None else "not_evaluated")
        return found, measured_samples, patch, weights_patch, winning_direction
    finally:
        region_growth._scalar_period_finish("period_measurement", reason, attempted_rounds=executed_rounds)


_LOCAL_PERIOD_WINDOWS = ((-1, -1), (-1, 1), (1, -1), (1, 1), (0, 0))
_LOCAL_PERIOD_MAX_CHILDREN = 40
_local_period_state = ContextVar("diagnostic_local_period", default=None)


@contextmanager
def _local_period_diagnostic_context(observer, *, enabled=False):
    """Evaluation-only opt-in; never changes a production reading or signature."""
    state = (dict(observer=observer, used=0, parents=set()) if enabled else None)
    token = _local_period_state.set(state)
    try:
        yield state
    finally:
        _local_period_state.reset(token)


def _local_period_proposals(
    image, axis, axis_matrix, focal, *, mask, centre, principal_point,
    extent, side, max_rectified_side, parent_identity,
):
    """Observe the five fixed windows of an already failed same-pass parent.

    No arrays or promoted readings escape this helper. The unchanged measurement
    loop may adjust sampling, but the five plane windows are fixed in advance.
    All child hooks use separate stages under the parent's scalar budget.
    """
    state = _local_period_state.get()
    if state is None:
        return
    observer = state["observer"]
    if observer is None or region_growth._attrition_state.get() is not observer:
        if observer is not None:
            observer.incomplete = True
        return
    parent_key = observer._encode(parent_identity).decode()
    parent_measurements = observer.stages.get("period_measurement", {})
    if parent_measurements.get("terminal", {}).get(parent_key) != "no_directional_period":
        return
    if parent_key in state["parents"]:
        observer.incomplete = True
        return
    state["parents"].add(parent_key)
    # The reading seam reserves this independently, so a wholly missing helper
    # cannot silently disappear. Direct synthetic callers may reserve it here.
    if parent_key not in observer.stages.get("local_period_parent", {}).get("expected", {}):
        observer.expect("local_period_parent", parent_identity)
    reason, completed = "exception", 0
    identities = tuple((*parent_identity, "local", i) for i in range(5))
    try:
        if (state["used"] + 5 > _LOCAL_PERIOD_MAX_CHILDREN or observer.incomplete):
            reason = "child_budget_exhausted"
            observer.incomplete = True
            return
        if (type(max_rectified_side) is not int or not 64 <= max_rectified_side <= MAX_RECTIFIED_SIDE
                or not np.isfinite((*centre, extent, side, focal)).all()
                or extent <= 0 or side not in (-1., 1.)):
            reason = "invalid_geometry_or_side_cap"
            observer.incomplete = True
            return
        child_extent = float(extent) / 2.
        requested = int(min(max(64, lattice.sampling_for(child_extent, focal)), max_rectified_side))
        # Census and reserve every window before even the first rectification.
        for identity in identities:
            observer.expect("local_period_child", identity)
            observer.expect("local_period_measurement", identity)
        if observer.incomplete or any(
            observer._encode(identity).decode() not in observer.stages[stage]["expected"]
            for identity in identities for stage in ("local_period_child", "local_period_measurement")
        ):
            reason = "accounting_budget_or_missing_expectation"
            observer.incomplete = True
            return
        state["used"] += 5
        for index, (du, dv) in enumerate(_LOCAL_PERIOD_WINDOWS):
            if observer.incomplete:
                reason = "accounting_budget_exhausted"
                return
            identity = identities[index]
            child_centre = (float(centre[0]) + du * child_extent,
                            float(centre[1]) + dv * child_extent)
            with region_growth._scalar_period_context(
                identity=identity, axis=int(axis), stage_prefix="local_",
            ):
                found, samples, patch, weights, direction = _bounded_measure(
                    image, axis, axis_matrix, focal, mask=mask, centre=child_centre,
                    principal_point=principal_point, extent=child_extent, side=side,
                    max_rectified_side=max_rectified_side)
            step = lattice.rectified_pixel_step(child_extent, samples)
            converted = (lattice.period_hypotheses(found, extent=child_extent, samples=samples)
                         if found is not None else None)
            child_key = observer._encode(identity).decode()
            disposition = observer.stages["local_period_measurement"]["terminal"].get(child_key)
            if disposition is None:
                observer.incomplete = True
                disposition = "missing_measurement_hook"
            observer.finish("local_period_child", identity, disposition,
                parent_identity=parent_identity, child_index=index, axis=int(axis),
                parent_reason="no_directional_period", parent_centre=tuple(map(float, centre)),
                parent_extent=float(extent), centre=child_centre, extent=child_extent, side=float(side),
                requested_samples=requested, measured_samples=int(samples), pixel_step=step,
                period_pixels=float(found.period_pixels) if found is not None else None,
                strength=float(found.strength) if found is not None else None,
                direction=direction,
                plane_unit_period=float(found.period_pixels) * step if found is not None else None,
                hypotheses=tuple(dict(interpretation=h.interpretation, blocks_per_period=h.blocks_per_period,
                    plane_distance=h.plane_distance, score=h.score) for h in converted.hypotheses)
                    if converted is not None else (),
                result_state="measured" if found is not None else "abstained")
            completed += 1
            # Drop rectified child buffers before measuring the next child.
            del patch, weights
        reason = "completed"
    finally:
        observer.finish("local_period_parent", parent_identity, reason,
            axis=int(axis), parent_reason="no_directional_period", planned_children=5,
            completed_children=completed, invocation_reserved_children=state["used"],
            max_children=_LOCAL_PERIOD_MAX_CHILDREN)
        expected_keys = {observer._encode(identity).decode() for identity in identities}
        child_stage = observer.stages.get("local_period_child", {})
        detail_rows = [v.get("details", {}) for v in child_stage.get("counts", {}).values()
                       if v.get("details", {}).get("parent_identity") == parent_identity]
        parent_stage = observer.stages.get("local_period_parent", {})
        parent_details = [v.get("details", {}) for v in parent_stage.get("counts", {}).values()
                          if v.get("identity") == parent_identity]
        closed = (reason == "completed" and completed == 5
                  and parent_key in parent_stage.get("expected", {})
                  and parent_stage.get("terminal", {}).get(parent_key) == "completed"
                  and len(parent_details) == 1
                  and expected_keys <= set(child_stage.get("expected", {}))
                  and expected_keys <= set(child_stage.get("terminal", {}))
                  and len(detail_rows) == 5
                  and {row["identity"] for row in detail_rows} == set(identities))
        observer.check("local_period_child_accounting",
            observer.checks.get("local_period_child_accounting", True) and closed)


def _autocorrelation_support(
    patch: np.ndarray,
    period: float,
    multiple: int,
    *,
    along: int = 0,
    weights: np.ndarray | None = None,
) -> float:
    """Normalised autocorrelation of the rectified patch at ``multiple`` periods.

    The same profile and normalisation :func:`lattice.estimate_period` uses for
    its sub-block ``block_supports``; here it is asked about the supra-block
    lags ``k * period`` that ``estimate_period`` deliberately suppresses. A
    ``0.0`` return means the lag fell outside the profile, not that the lag was
    tested and failed.
    """
    with lattice._period_support_context(profile_use="supra", multiple=multiple, period_direction=along):
        try:
            if along not in (0, 1) or not np.isfinite(period) or period <= 0.0 or multiple <= 0:
                return 0.0
            try:
                signal = lattice._profile(np.asarray(patch), along, weights)
            except (TypeError, ValueError, IndexError):
                return 0.0
            signal = signal - signal.mean()
            if signal.size < 8 or float(signal.var()) < lattice.MINIMUM_CONTRAST:
                return 0.0
            correlation = np.correlate(signal, signal, mode="full")[signal.size - 1:]
            if correlation[0] <= 0.0:
                return 0.0
            correlation = correlation / correlation[0]
            index = int(round(period * multiple))
            if index <= 0 or index >= correlation.size:
                return 0.0
            return float(max(0.0, correlation[index]))

        finally:
            lattice._period_support_event("supra_support", multiple=multiple, along=along,
                period=period, signal=locals().get("signal"),
                correlation=locals().get("correlation"), lag=locals().get("index"))


def _distance_alternatives(
    estimate: "lattice.PeriodEstimate",
    *,
    extent: float,
    samples: int,
    patch: np.ndarray,
    weights: np.ndarray | None = None,
    profile_direction: int = 0,
) -> tuple[DistanceAlternative, ...]:
    """Build ``B`` from the measured period, then prune on declared support only.

    Support provenance: ``b = 1`` is the reference reading and carries 1.0;
    ``b = 1/m`` for ``m`` in ``estimate.block_supports`` (2/4/8/16) carries that
    correlation; ``b = k`` for ``k`` in 2/3/4/8/16 carries the normalised
    autocorrelation at lag ``k * period``. This supra-block support uses the
    winning profile axis and the same rectified proposal weights as the period
    estimate. Sorting is by support descending, then
    ``b`` descending, and only ``support >= SUPPORT_PRUNE * max(support)``
    survives — the preregistered floor.
    """
    if (not np.isfinite((estimate.period_pixels, extent)).all()
            or estimate.period_pixels <= 0.0 or extent <= 0.0
            or not isinstance(samples, (int, np.integer)) or isinstance(samples, (bool, np.bool_))
            or samples < 2):
        return ()
    try:
        world = estimate.period_pixels * lattice.rectified_pixel_step(extent, samples)
    except ValueError:
        return ()
    if world <= 0.0 or not np.isfinite(world):
        return ()
    supports: dict[float, float] = {1.0: 1.0}
    for multiple, support in estimate.block_supports:
        if multiple > 0 and np.isfinite(support) and support >= 0.0:
            supports[1.0 / float(multiple)] = float(support)
    direction = profile_direction if profile_direction in (0, 1) else 0
    for multiple in (2, 3, 4, 8, 16):
        supports[float(multiple)] = _autocorrelation_support(
            patch, estimate.period_pixels, multiple, along=direction, weights=weights)
    ranked = sorted(supports.items(), key=lambda item: (-item[1], -item[0]))
    if not ranked:
        return ()
    floor = SUPPORT_PRUNE * ranked[0][1]
    alternatives = []
    for b, support in ranked:
        distance = float(b / world)
        if (support >= floor and np.isfinite((b, distance, support)).all()
                and b > 0.0 and distance > 0.0 and support >= 0.0):
            alternatives.append(DistanceAlternative(float(b), distance, float(support)))
    lattice._period_support_event("distance_alternatives", estimate=estimate,
        extent=extent, samples=samples, profile_direction=direction, world_period=world,
        evaluated=tuple(supports.items()), ranked=tuple(ranked), prune_floor=floor,
        prune_share=SUPPORT_PRUNE, retained=tuple(alternatives))
    return tuple(alternatives)


def _folded_repeat_fraction(profile: np.ndarray, period: float, *, bins: int = STRUCTURAL_FOLD_BINS) -> float:
    """Share of a 1-D edge profile consistent with one lattice of ``period``.

    The profile is folded modulo the period and histogrammed; the widest cyclic
    window of ``supported.PHASE_TOLERANCE`` of the period is the energy that one
    lattice phase explains. A uniform wall of identical blocks — block borders
    and even its interior texture lines — folds onto the same few positions every
    period and scores near one; an opening or a material boundary folds
    everywhere and scores lower.
    """
    total = float(profile.sum())
    if total <= 0.0 or period < 2.0:
        return 0.0
    index = np.arange(profile.size, dtype=float)
    folded_index = np.mod(index, period) / period
    binned = np.clip((folded_index * bins).astype(int), 0, bins - 1)
    folded = np.bincount(binned, weights=profile, minlength=bins)
    window = max(1, int(round(supported.PHASE_TOLERANCE * bins)))
    extended = np.concatenate((folded, folded[:window]))
    cumulative = np.concatenate(([0.0], np.cumsum(extended)))
    windowed = cumulative[window:window + bins] - cumulative[:bins]
    return float(windowed.max() / total)


def _structural_share(weights_patch: np.ndarray, period_pixels: float) -> float:
    """Fraction of a region's edge weight *not* explained by its own lattice.

    Each in-plane profile is folded at the measured period and the largest
    single-phase window is the energy one lattice explains; the share is one
    minus the better of the two directions. The perpendicular seams that cross
    every column (or row) are a constant baseline and are removed before the
    fold, otherwise they would dominate a profile that is supposed to measure
    the seams running *along* it. A plane whose content is one uniform repeating
    texture scores near zero and is ineligible for identity; a plane carrying an
    opening or an aperiodic material change keeps energy at phases no single
    lattice explains and scores higher. The ``min_structural_share`` gate in the
    graph is the primary repetition test; the correlation peak-to-sidelobe ratio
    is a scored cue beside it.
    """
    patch = np.asarray(weights_patch, dtype=float)
    if patch.ndim != 2 or float(patch.sum()) <= 0.0 or period_pixels < 2.0:
        return 0.0
    across = _baseline_removed(patch.sum(axis=0))
    down = _baseline_removed(patch.sum(axis=1))
    repeating = max(
        _folded_repeat_fraction(across, period_pixels),
        _folded_repeat_fraction(down, period_pixels),
    )
    return float(np.clip(1.0 - repeating, 0.0, 1.0))


def _baseline_removed(profile: np.ndarray) -> np.ndarray:
    """Drop the constant a profile inherits from the perpendicular seams."""
    values = np.asarray(profile, dtype=float)
    if values.size == 0:
        return values
    return np.clip(values - float(np.median(values)), 0.0, None)


def _periodic_energy_axis(
    profile: np.ndarray | None,
    period_pixels: float,
    *,
    source_finite: bool = True,
) -> tuple[float | None, float | None, float | None, float | None, str | None]:
    """One shared legacy/evidence phase calculation with explicit abstention."""
    if not source_finite or profile is None:
        return None, None, None, None, "nonfinite_profile"
    values = np.asarray(profile, dtype=float).reshape(-1)
    if not np.all(np.isfinite(values)):
        return None, None, None, None, "nonfinite_profile"
    if np.any(values < 0.0):
        return None, None, None, None, "invalid_profile"
    if values.size == 0:
        return None, 0.0, 0.0, 0.0, "empty_profile"
    mass = float(values.sum())
    if not np.isfinite(mass):
        return None, None, None, None, "nonfinite_profile"
    if mass <= 1e-9:
        return None, 0.0, mass, 0.0, "empty_profile"
    if not np.isfinite(period_pixels) or period_pixels <= 0.0:
        return None, None, mass, None, "invalid_period"
    offsets = np.arange(values.size, dtype=float)
    response = supported._periodic_energy_resultant(
        values, offsets, float(period_pixels))
    magnitude = float(abs(response))
    if not np.isfinite(magnitude):
        return None, None, mass, None, "nonfinite_profile"
    concentration = float(np.clip(magnitude / mass, 0.0, 1.0))
    if period_pixels < 2.0:
        return None, magnitude, mass, concentration, "insufficient_period"
    if magnitude <= 1e-12:
        return None, magnitude, mass, concentration, "zero_resultant"
    phase = float((-np.angle(response) / (2.0 * np.pi)
                   * period_pixels) % period_pixels)
    # The circular mean can return period-epsilon for a true zero phase due to
    # floating-point angle noise. Canonicalize only that numerical endpoint.
    if phase >= period_pixels - max(1e-12, period_pixels * 1e-12):
        phase = 0.0
    return phase, magnitude, mass, concentration, None


def _periodic_energy_measurements(
    weights_patch: np.ndarray, period_pixels: float,
) -> tuple[tuple[float | None, float | None, float | None, float | None, str | None],
           tuple[float | None, float | None, float | None, float | None, str | None]]:
    """Shared structured (u, v) periodic-energy primitive for old and new APIs."""
    try:
        patch = np.asarray(weights_patch)
        finite = patch.ndim == 2 and bool(np.all(np.isfinite(patch)))
    except (TypeError, ValueError, OverflowError):
        patch = np.empty((0, 0), dtype=float)
        finite = False
    profiles: tuple[np.ndarray | None, np.ndarray | None]
    if finite:
        patch = np.asarray(patch, dtype=float)
        profiles = (patch.sum(axis=0), patch.sum(axis=1))
    else:
        profiles = (None, None)
    return tuple(_periodic_energy_axis(profile, float(period_pixels),
                                       source_finite=finite)
                 for profile in profiles)


def _periodic_energy_phase_evidence(
    weights_patch: np.ndarray,
    period_pixels: float,
    *,
    centre: tuple[float, float],
    extent: float,
    samples: int,
    source_region_id: str | None = None,
) -> PeriodicEnergyPhaseEvidence:
    """Measure raw periodic-energy phase and retain its exact patch context."""
    measured = _periodic_energy_measurements(weights_patch, period_pixels)
    return PeriodicEnergyPhaseEvidence(
        phase_uv_pixels=(measured[0][0], measured[1][0]),
        period_pixels=float(period_pixels),
        patch_centre_uv=(float(centre[0]), float(centre[1])),
        extent=float(extent),
        samples=int(samples),
        response_magnitude_uv=(measured[0][1], measured[1][1]),
        profile_mass_uv=(measured[0][2], measured[1][2]),
        concentration_uv=(measured[0][3], measured[1][3]),
        unavailable_reason_uv=(measured[0][4], measured[1][4]),
        source_region_id=source_region_id,
    )


def _phase_pixels(weights_patch: np.ndarray, period_pixels: float) -> tuple[float, float]:
    """Legacy zero-sentinel view of the structured periodic-energy evidence."""
    result = tuple(axis[0] for axis in
                   _periodic_energy_measurements(weights_patch, period_pixels))
    return tuple(float(value if value is not None else 0.0)
                 for value in result)


def _weight_image(segments: np.ndarray, weights: np.ndarray, height: int, width: int) -> np.ndarray:
    """Rasterise per-segment structural weight into a source-resolution image."""
    import cv2

    out = np.zeros((height, width), dtype=np.float32)
    if not len(segments):
        return out
    order = np.lexsort((
        segments[:, 3], segments[:, 2], segments[:, 1], segments[:, 0],
    ))
    for index in order:
        x0, y0, x1, y1 = (float(value) for value in segments[index, :4])
        weight = float(weights[index]) if index < len(weights) else 0.0
        if weight <= 0.0:
            continue
        cv2.line(out, (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))),
                 weight, 1, lineType=cv2.LINE_8)
    return out


def frame_plane_readings(
    image: np.ndarray,
    usable: np.ndarray,
    calibration_result: CalibrationResult,
    *,
    frame: str = "",
    max_readings: int = MAX_READINGS,
    min_structural_share: float = MIN_STRUCTURAL_SHARE,
    max_rectified_side: int = MAX_RECTIFIED_SIDE,
    diagnostics: dict | None = None,
) -> tuple[PlaneReading, ...]:
    """Candidate plane readings for one frame, from pixels and the usable mask.

    ``frame`` is the stable frame id the readings carry; it is a keyword because
    :class:`~mc_photogrammetry.reconstruction.calibration.CalibrationResult`
    deliberately records no frame identity. ``usable`` is True where a pixel may
    be used (``~create_gui_mask(h, w)``); excluded pixels are zeroed before
    detection and every region mask is intersected with it.

    Returns up to ``max_readings`` readings, ordered by ``(axis, side,
    -confidence, polygon)``, with at most
    :data:`MAX_READINGS_PER_AXIS_SIDE` per ``(axis, side)``. Readings below
    ``min_structural_share`` are retained (so the graph can refuse identity) but
    rank below eligible ones and cannot use up the cap ahead of them.
    """
    usable = np.asarray(usable, dtype=bool)
    if usable.ndim != 2:
        raise ValueError("usable must be a two-dimensional boolean mask")
    if image.shape[:2] != usable.shape:
        raise ValueError("image and usable mask must share a shape")
    if calibration_result.width != usable.shape[1] or calibration_result.height != usable.shape[0]:
        raise ValueError("calibration_result dimensions do not match the frame")

    diagnostics = {} if diagnostics is None else diagnostics
    diagnostics.setdefault("abstentions", [])
    diagnostics.setdefault("segments", 0)
    diagnostics.setdefault("low_structural_share", 0)

    grey = _masked_grey(image, usable)
    segments = detect_segments(enhance(grey))
    diagnostics["segments"] = int(len(segments))
    if len(segments) < MINIMUM_SEGMENTS:
        diagnostics["abstentions"].append(
            ReadingAbstention(frame, None, None, "no_plane_region",
                              f"only {len(segments)} segments detected"))
        return ()

    weights = structural_weights(grey, segments)
    axis_matrix = np.asarray(
        axis_directions(calibration_result.yaw_degrees, calibration_result.pitch_degrees,
                        calibration_result.roll_degrees),
        dtype=float,
    )
    focal = float(calibration_result.focal_pixels)
    principal = calibration_result.principal_point or (
        calibration_result.width / 2.0, calibration_result.height / 2.0)
    principal = (float(principal[0]), float(principal[1]))
    height, width = usable.shape

    weight_image = _weight_image(segments, weights, height, width)
    lattice._period_support_event("frame_structural_source", frame=frame,
        segments=segments, segment_weights=weights, calibration=calibration_result,
        grey=grey, usable=usable, structural_weights=weight_image)
    candidate_regions = _candidate_regions(
        grey, usable, calibration_result, segments, weights, diagnostics)
    if not candidate_regions:
        diagnostics["abstentions"].append(
            ReadingAbstention(frame, None, None, "no_plane_region",
                              "neither proposal source produced a region"))
        return ()
    candidates = [(*candidate, None) for candidate in candidate_regions]

    return _measure_plane_candidates(
        grey, usable, weight_image, candidates, axis_matrix, focal, principal,
        frame=frame, max_readings=max_readings,
        min_structural_share=min_structural_share,
        max_rectified_side=max_rectified_side, diagnostics=diagnostics,
    )


def _measure_plane_candidates(
    grey: np.ndarray,
    usable: np.ndarray,
    weight_image: np.ndarray,
    candidates: Sequence[tuple[
        int, np.ndarray, tuple[tuple[float, float], ...], float,
        PlaneRegionEvidence | None,
    ]],
    axis_matrix: np.ndarray,
    focal: float,
    principal: tuple[float, float],
    *,
    frame: str,
    max_readings: int,
    min_structural_share: float,
    max_rectified_side: int,
    diagnostics: dict,
) -> tuple[PlaneReading, ...]:
    """The shared measurement, ranking, and cap path for image plane masks."""
    lattice._period_support_event("frame_measure_source", frame=frame, grey=grey,
        usable=usable, structural_weights=weight_image, axis_matrix=axis_matrix,
        focal=focal, principal_point=principal,
        selection_policy=dict(min_structural_share=min_structural_share,
            max_readings=max_readings,max_per_axis_side=MAX_READINGS_PER_AXIS_SIDE,
            max_rectified_side=max_rectified_side))
    capture_candidate_ids = {} if lattice._period_support_state() is not None else None
    observer = region_growth._attrition_state.get()
    observed_readings = {} if observer is not None else None
    tentative: list[tuple[bool, PlaneReading]] = []
    for candidate_ordinal, (axis, region_mask, polygon, _score, region_evidence) in enumerate(candidates):
        observation_id = None
        if observer is not None:
            with region_growth._scalar_observation_guard(observer):
                observation_id = observer.bindings.get(("candidate", frame, id(region_mask)))
                if observation_id is None:
                    observation_id = (frame, candidate_ordinal)
                    observer.expect("reading", observation_id)
        with lattice._period_support_context(frame=str(frame), candidate_id=(str(frame), candidate_ordinal), candidate_axis=int(axis)):
            lattice._period_support_event("candidate_begin", axis=axis, polygon=polygon,
                proposal_strength=_score, region_evidence=region_evidence)
            window = lattice.plane_window(region_mask, axis, axis_matrix, focal,
                                          principal_point=principal, margin=1.05)
            if window is None:
                if observer is not None:
                    with region_growth._scalar_observation_guard(observer):
                        observer.finish("reading", observation_id, "window_incidence", axis=int(axis))
                diagnostics["abstentions"].append(
                    ReadingAbstention(frame, int(axis), None, "unmeasurable_plane",
                                      "plane_window found no usable ray incidence"))
                continue
            if (len(window) != 4 or not np.isfinite(window).all()
                    or float(window[2]) <= 0.0 or float(window[3]) not in (-1.0, 1.0)):
                if observer is not None:
                    with region_growth._scalar_observation_guard(observer):
                        observer.finish("reading", observation_id, "window_invalid", axis=int(axis))
                diagnostics["abstentions"].append(
                    ReadingAbstention(frame, int(axis), None, "unmeasurable_plane",
                                      "plane_window returned invalid coordinates or extent"))
                continue
            lattice._period_support_event("candidate_window", window=window, region_evidence=region_evidence)
            centre = (float(window[0]), float(window[1]))
            extent = float(window[2])
            side = float(window[3])
            with region_growth._scalar_period_context(identity=observation_id, axis=int(axis)):
                region_growth._scalar_period_expect("period_measurement")
                found, samples, patch, weights_patch, profile_direction = _bounded_measure(
                    grey, axis, axis_matrix, focal, mask=region_mask, centre=centre,
                    principal_point=principal, extent=extent, side=side,
                    max_rectified_side=int(max_rectified_side))
            if found is None or patch is None or weights_patch is None:
                if observer is not None:
                    with region_growth._scalar_observation_guard(observer):
                        observer.finish("reading", observation_id, "periodicity", axis=int(axis), side=side)
                    if _local_period_state.get() is not None:
                        with region_growth._scalar_observation_guard(observer):
                            parent_key = observer._encode(observation_id).decode()
                            if observer.stages.get("period_measurement", {}).get("terminal", {}).get(
                                parent_key) == "no_directional_period":
                                observer.expect("local_period_parent", observation_id)
                                _local_period_proposals(grey, axis, axis_matrix, focal,
                                    mask=region_mask, centre=centre, principal_point=principal,
                                    extent=extent, side=side, max_rectified_side=int(max_rectified_side),
                                    parent_identity=observation_id)
                diagnostics["abstentions"].append(
                    ReadingAbstention(frame, int(axis), side, "unmeasurable_plane",
                                      "no periodic lattice on the proposed region"))
                continue
            # The edge-energy patch is the rasterised per-segment structural weight
            # rectified through the *same* window: the phase and the structural share
            # both need where the contrast-weighted lines land, not where the region
            # mask lands.
            edge_patch = lattice.rectify_plane(
                weight_image, axis, axis_matrix, focal, extent=extent, samples=samples,
                centre=centre, principal_point=principal, side=side)
            if edge_patch is None or float(edge_patch.sum()) <= 0.0:
                if observer is not None:
                    with region_growth._scalar_observation_guard(observer):
                        observer.finish("reading", observation_id, "structural_energy", axis=int(axis), side=side)
                diagnostics["abstentions"].append(
                    ReadingAbstention(frame, int(axis), side, "unmeasurable_plane",
                                      "no structural edge weight inside the region"))
                continue
            alternatives = _distance_alternatives(
                found, extent=extent, samples=samples, patch=patch, weights=weights_patch,
                profile_direction=profile_direction if profile_direction is not None else 0)
            if not alternatives:
                if observer is not None:
                    with region_growth._scalar_observation_guard(observer):
                        observer.finish("reading", observation_id, "distance_alternative", axis=int(axis), side=side)
                diagnostics["abstentions"].append(
                    ReadingAbstention(frame, int(axis), side, "unmeasurable_plane",
                                      "no distance alternative survived the support floor"))
                continue
            share = _structural_share(edge_patch, found.period_pixels)
            coverage_cells = int(samples // max(found.period_pixels, 1.0))
            if coverage_cells < 1:
                if observer is not None:
                    with region_growth._scalar_observation_guard(observer):
                        observer.finish("reading", observation_id, "coverage", axis=int(axis), side=side,
                                        coverage_cells=coverage_cells)
                diagnostics["abstentions"].append(
                    ReadingAbstention(frame, int(axis), side, "insufficient_coverage",
                                      "the window spans less than one measured period"))
                continue
            eligible = share >= float(min_structural_share)
            if not eligible:
                diagnostics["low_structural_share"] += 1
            coverage_term = min(coverage_cells / 8.0, 1.0)
            strength_term = min(max(float(found.strength), 0.0) / 0.5, 1.0)
            share_term = min(share / 0.5, 1.0)
            confidence = float(np.clip(0.4 * share_term + 0.3 * coverage_term + 0.3 * strength_term, 0.0, 1.0))
            phase_evidence = _periodic_energy_phase_evidence(
                edge_patch, found.period_pixels, centre=centre, extent=extent,
                samples=samples,
                source_region_id=(region_evidence.source_region_id
                                  if region_evidence is not None else None),
            )
            reading = PlaneReading(
                frame=str(frame),
                axis=int(axis),
                side=side,
                alternatives=alternatives,
                period_pixels=float(found.period_pixels),
                phase_pixels=tuple(float(value if value is not None else 0.0)
                                   for value in phase_evidence.phase_uv_pixels),
                extent=extent,
                samples=int(samples),
                polygon=tuple((float(x), float(y)) for x, y in polygon),
                coverage_cells=coverage_cells,
                structural_share=share,
                confidence=confidence,
                rectification=PlaneRectification(
                    axis_matrix=tuple(tuple(float(value) for value in row) for row in axis_matrix),
                    focal=focal,
                    principal_point=principal,
                    centre=centre,
                ),
                region_evidence=region_evidence,
                phase_evidence=phase_evidence,
            )
            tentative.append((eligible, reading))
            if observer is not None:
                with region_growth._scalar_observation_guard(observer):
                    observed_readings[id(reading)] = observation_id
            if capture_candidate_ids is not None:
                capture_candidate_ids[id(reading)] = (str(frame), candidate_ordinal)
            lattice._period_support_event("candidate_reading", reading=reading, eligible=eligible,
                structural_energy_patch=edge_patch, confidence_terms=(coverage_term, strength_term, share_term))

    if not tentative:
        return ()

    tentative.sort(key=lambda item: (
        not item[0],
        item[1].axis,
        item[1].side,
        -item[1].confidence,
        item[1].polygon or (),
        (item[1].region_evidence.source_region_id
         if item[1].region_evidence is not None else ""),
    ))
    # Give each available axis/side one opportunity before a second reading
    # from any bucket. Eligibility still takes precedence over all ineligible
    # retained readings, and each bucket keeps the ranking above unchanged.
    buckets: dict[tuple[bool, int, float], list[PlaneReading]] = {}
    for eligible, reading in tentative:
        buckets.setdefault((eligible, reading.axis, reading.side), []).append(reading)
    per_axis_side: dict[tuple[int, float], int] = {}
    selected: list[PlaneReading] = []
    for eligible in (True, False):
        keys = sorted(key for key in buckets if key[0] == eligible)
        round_index = 0
        while any(round_index < len(buckets[key]) for key in keys):
            for bucket_key in keys:
                if round_index >= len(buckets[bucket_key]):
                    continue
                reading = buckets[bucket_key][round_index]
                key = (reading.axis, reading.side)
                if per_axis_side.get(key, 0) >= MAX_READINGS_PER_AXIS_SIDE:
                    if observer is not None:
                        with region_growth._scalar_observation_guard(observer):
                            observer.finish("reading", observed_readings[id(reading)], "per_axis_side_cap",
                                            axis=int(reading.axis), side=float(reading.side))
                    continue
                per_axis_side[key] = per_axis_side.get(key, 0) + 1
                selected.append(reading)
                if len(selected) >= int(max_readings):
                    break
            if len(selected) >= int(max_readings):
                break
            round_index += 1
        if len(selected) >= int(max_readings):
            break
    selected.sort(key=lambda reading: (reading.axis, reading.side, -reading.confidence,
                                       reading.polygon or (),
                                       (reading.region_evidence.source_region_id
                                        if reading.region_evidence is not None else "")))
    diagnostics["readings"] = len(selected)
    if observer is not None:
        with region_growth._scalar_observation_guard(observer):
            selected_ids = {id(reading) for reading in selected}
            canonical = sorted(selected, key=reading_order_key)
            plane_indices = {id(reading): ordinal for ordinal, reading in enumerate(canonical)}
            for _eligible, reading in tentative:
                identity = observed_readings[id(reading)]
                if id(reading) in selected_ids:
                    observer.finish("reading", identity, "accepted", axis=int(reading.axis),
                                    side=float(reading.side), plane_index=plane_indices[id(reading)],
                                    coverage_cells=int(reading.coverage_cells),
                                    structural_share=float(reading.structural_share),
                                    confidence=float(reading.confidence))
                    observer.bind(("reading", id(reading)), (frame, plane_indices[id(reading)]))
                elif observer._encode(identity).decode() not in observer._stage("reading")["terminal"]:
                    observer.finish("reading", identity, "reading_cap", axis=int(reading.axis),
                                    side=float(reading.side))
    if capture_candidate_ids is not None:
        lattice._period_support_event("frame_reading_selection", frame=frame,
            ranked=tuple((capture_candidate_ids[id(reading)], eligible, reading)
                         for eligible, reading in tentative),
            selected=tuple((capture_candidate_ids[id(reading)], reading) for reading in selected),
            max_readings=max_readings, max_per_axis_side=MAX_READINGS_PER_AXIS_SIDE,
            min_structural_share=min_structural_share,max_rectified_side=max_rectified_side)
    return tuple(selected)


def _proposal_mask_candidates(
    proposal_result: boundary_candidates.PlaneProposalResult,
    usable: np.ndarray,
    frame: str,
    diagnostics: dict,
) -> list[tuple[
    int, np.ndarray, tuple[tuple[float, float], ...], float, PlaneRegionEvidence,
]]:
    """Validate/canonicalize proposal masks without deduplicating their regions.

    Distinct regions survive even when their normal and ray side match. The
    shared measurement loop applies its existing cap after measuring each one;
    this adapter never groups by ``(axis, side)`` and derives side anew through
    :func:`lattice.plane_window`.
    """
    height, width = usable.shape
    candidates = []
    observer = region_growth._attrition_state.get()
    for proposal_ordinal, proposal in enumerate(proposal_result.proposals):
        observation_id = (frame, proposal_ordinal)
        if not isinstance(proposal, boundary_candidates.PlaneProposal):
            diagnostics["abstentions"].append(
                ReadingAbstention(frame, None, None, "unmeasurable_plane",
                                  "proposal entry has an invalid type"))
            if observer is not None:
                with region_growth._scalar_observation_guard(observer):
                    refusal = diagnostics["abstentions"][-1]
                    observer.finish("reading", observation_id, "proposal_validation",
                                    axis=refusal.axis, typed_reason=refusal.reason, detail=refusal.detail)
            continue
        axis = (int(proposal.axis)
                if isinstance(proposal.axis, (int, np.integer))
                and not isinstance(proposal.axis, (bool, np.bool_))
                and int(proposal.axis) in (0, 1, 2) else None)
        if axis is None:
            diagnostics["abstentions"].append(
                ReadingAbstention(frame, None, None, "unmeasurable_plane",
                                  "proposal axis must be 0, 1, or 2"))
            if observer is not None:
                with region_growth._scalar_observation_guard(observer):
                    refusal = diagnostics["abstentions"][-1]
                    observer.finish("reading", observation_id, "proposal_validation",
                                    axis=refusal.axis, typed_reason=refusal.reason, detail=refusal.detail)
            continue
        try:
            raw_mask = np.asarray(proposal.mask)
        except (TypeError, ValueError):
            diagnostics["abstentions"].append(
                ReadingAbstention(frame, axis, None, "unmeasurable_plane",
                                  "proposal mask is not a rectangular array"))
            if observer is not None:
                with region_growth._scalar_observation_guard(observer):
                    refusal = diagnostics["abstentions"][-1]
                    observer.finish("reading", observation_id, "proposal_validation",
                                    axis=refusal.axis, typed_reason=refusal.reason, detail=refusal.detail)
            continue
        if (raw_mask.ndim != 2 or raw_mask.shape != (height, width)
                or not (np.issubdtype(raw_mask.dtype, np.number)
                        or np.issubdtype(raw_mask.dtype, np.bool_))
                or (np.issubdtype(raw_mask.dtype, np.number)
                    and not np.isfinite(raw_mask).all())):
            diagnostics["abstentions"].append(
                ReadingAbstention(frame, axis, None, "unmeasurable_plane",
                                  "proposal mask must be finite and match image dimensions"))
            if observer is not None:
                with region_growth._scalar_observation_guard(observer):
                    refusal = diagnostics["abstentions"][-1]
                    observer.finish("reading", observation_id, "proposal_validation",
                                    axis=refusal.axis, typed_reason=refusal.reason, detail=refusal.detail)
            continue
        region_mask = ((raw_mask > 0.5) & usable).astype(np.float32)
        binary_mask = region_mask > 0.5
        row_counts = np.count_nonzero(binary_mask, axis=1)
        column_counts = np.count_nonzero(binary_mask, axis=0)
        occupied_rows = np.flatnonzero(row_counts)
        occupied_columns = np.flatnonzero(column_counts)
        area = int(row_counts.sum())
        if area < 3 or len(occupied_rows) < 2 or len(occupied_columns) < 2:
            diagnostics["abstentions"].append(
                ReadingAbstention(frame, axis, None, "unmeasurable_plane",
                                  "proposal mask is empty or geometrically degenerate"))
            if observer is not None:
                with region_growth._scalar_observation_guard(observer):
                    refusal = diagnostics["abstentions"][-1]
                    observer.finish("reading", observation_id, "proposal_validation",
                                    axis=refusal.axis, typed_reason=refusal.reason, detail=refusal.detail)
            continue
        y0 = int(occupied_rows[0])
        row_x = np.flatnonzero(binary_mask[y0])
        x0 = int(row_x[0])
        if len(row_x) > 1:
            x1, y1 = int(row_x[-1]), y0
        else:
            y1 = int(occupied_rows[1])
            next_x = np.flatnonzero(binary_mask[y1])
            x1 = int(next_x[0])
        dx, dy = x1 - x0, y1 - y0
        non_collinear = False
        for row in occupied_rows:
            xs_in_row = np.flatnonzero(binary_mask[row])
            endpoints = (int(xs_in_row[0]), int(xs_in_row[-1]))
            if any(dx * (int(row) - y0) - dy * (x - x0) != 0 for x in endpoints):
                non_collinear = True
                break
        if not non_collinear:
            diagnostics["abstentions"].append(
                ReadingAbstention(frame, axis, None, "unmeasurable_plane",
                                  "proposal mask is empty or geometrically degenerate"))
            if observer is not None:
                with region_growth._scalar_observation_guard(observer):
                    refusal = diagnostics["abstentions"][-1]
                    observer.finish("reading", observation_id, "proposal_validation",
                                    axis=refusal.axis, typed_reason=refusal.reason, detail=refusal.detail)
            continue
        try:
            polygon = tuple((float(point[0]), float(point[1]))
                            for point in proposal.polygon)
            strength = float(proposal.strength)
            if (not isinstance(proposal.sources, (tuple, list))
                    or any(not isinstance(source, str) for source in proposal.sources)):
                raise ValueError("proposal sources must be a string sequence")
            sources = tuple(proposal.sources)
        except (TypeError, ValueError, IndexError):
            polygon, strength, sources = (), float("nan"), ()
        if (len(polygon) < 3
                or any(not np.isfinite(value) for point in polygon for value in point)
                or not np.isfinite(strength)):
            diagnostics["abstentions"].append(
                ReadingAbstention(frame, axis, None, "unmeasurable_plane",
                                  "proposal polygon or strength is invalid"))
            if observer is not None:
                with region_growth._scalar_observation_guard(observer):
                    refusal = diagnostics["abstentions"][-1]
                    observer.finish("reading", observation_id, "proposal_validation",
                                    axis=refusal.axis, typed_reason=refusal.reason, detail=refusal.detail)
            continue
        bbox = (int(occupied_rows[0]), int(occupied_columns[0]),
                int(occupied_rows[-1]), int(occupied_columns[-1]))
        sum_y = int(np.dot(np.arange(height, dtype=np.int64), row_counts))
        sum_x = int(np.dot(np.arange(width, dtype=np.int64), column_counts))
        mask_digest = hashlib.sha256(
            np.ascontiguousarray(binary_mask).tobytes()).digest()
        sort_key = (
            axis, bbox, sum_y, sum_x, area, polygon,
            -strength, sources, mask_digest,
        )
        try:
            evidence = _make_region_evidence(
                binary_mask, axis, polygon, sources)
        except (TypeError, ValueError, OverflowError):
            diagnostics["abstentions"].append(
                ReadingAbstention(frame, axis, None, "unmeasurable_plane",
                                  "proposal source-region evidence is invalid"))
            if observer is not None:
                with region_growth._scalar_observation_guard(observer):
                    refusal = diagnostics["abstentions"][-1]
                    observer.finish("reading", observation_id, "proposal_validation",
                                    axis=refusal.axis, typed_reason=refusal.reason, detail=refusal.detail)
            continue
        sort_key = (*sort_key, evidence.source_region_id)
        if observer is not None:
            with region_growth._scalar_observation_guard(observer):
                observer.bind(("candidate", frame, id(region_mask)), observation_id)
        candidates.append((sort_key, (axis, region_mask, polygon, strength, evidence)))
    candidates.sort(key=lambda item: item[0])
    return [candidate for _key, candidate in candidates]


def frame_plane_readings_from_proposals(
    image: np.ndarray,
    usable: np.ndarray,
    calibration_result: CalibrationResult,
    proposal_result: boundary_candidates.PlaneProposalResult,
    *,
    frame: str = "",
    max_readings: int = MAX_READINGS,
    min_structural_share: float = MIN_STRUCTURAL_SHARE,
    max_rectified_side: int = MAX_RECTIFIED_SIDE,
    diagnostics: dict | None = None,
) -> tuple[PlaneReading, ...]:
    """Measure v2 image-only region proposals with the existing lattice reader.

    Inputs are RGB pixels, the caller-derived usable mask, image-derived
    Manhattan calibration, and a typed proposal result. The proposals' masks
    are intersected with ``usable`` into new arrays before any measurement. No
    camera pose, capture manifest, ground truth, or reference fields are
    accepted here. Proposal axes are the C4 image-derived labels; proposal side
    metadata is ignored and the geometric side is recomputed from the mask.
    Invalid inputs become :class:`ReadingAbstention` values from the existing
    reason vocabulary instead of exceptions.
    """
    diagnostics = {} if diagnostics is None else diagnostics
    diagnostics.setdefault("abstentions", [])
    diagnostics.setdefault("segments", 0)
    diagnostics.setdefault("low_structural_share", 0)

    observer = region_growth._attrition_state.get()
    if observer is not None and isinstance(proposal_result, boundary_candidates.PlaneProposalResult):
        with region_growth._scalar_observation_guard(observer):
            for ordinal, _proposal in enumerate(proposal_result.proposals):
                observer.expect("reading", (str(frame), ordinal))

    def abstain(reason: str, detail: str, axis: int | None = None) -> tuple[PlaneReading, ...]:
        diagnostics["abstentions"].append(
            ReadingAbstention(frame, axis, None, reason, detail))
        if observer is not None:
            with region_growth._scalar_observation_guard(observer):
                for identity in observer.pending("reading", str(frame)):
                    observer.finish("reading", identity, "frame_rejection", axis=axis,
                                    typed_reason=reason, detail=detail)
        return ()

    if (not isinstance(image, np.ndarray) or image.ndim != 3
            or image.shape[2] != 3 or image.shape[0] <= 0 or image.shape[1] <= 0):
        return abstain("no_plane_region", "image must be a non-empty RGB array")
    try:
        usable_mask = np.asarray(usable, dtype=bool)
    except (TypeError, ValueError):
        return abstain("no_plane_region", "usable mask is invalid")
    if usable_mask.shape != image.shape[:2]:
        return abstain("no_plane_region", "image and usable mask dimensions differ")
    if not usable_mask.any():
        return abstain("no_plane_region", "usable mask contains no pixels")
    if not isinstance(calibration_result, CalibrationResult):
        return abstain("no_plane_region", "calibration result is missing or invalid")
    try:
        numeric = (calibration_result.focal_pixels, calibration_result.yaw_degrees,
                   calibration_result.pitch_degrees, calibration_result.roll_degrees)
        numeric_valid = all(np.isfinite(value) for value in numeric)
        principal = calibration_result.principal_point
        principal_valid = (principal is None or (
            len(principal) == 2 and all(np.isfinite(value) for value in principal)))
        focal_valid = calibration_result.focal_pixels > 0.0
        dimensions_match = ((calibration_result.width, calibration_result.height) ==
                            (image.shape[1], image.shape[0]))
    except (TypeError, ValueError):
        numeric_valid = principal_valid = focal_valid = dimensions_match = False
    if not (numeric_valid and principal_valid and focal_valid and dimensions_match):
        return abstain("no_plane_region", "calibration is invalid or dimension-mismatched")
    if not isinstance(proposal_result, boundary_candidates.PlaneProposalResult):
        return abstain("no_plane_region", "proposal result has an invalid type")
    if proposal_result.status != "ok" or not proposal_result.proposals:
        reason = str(proposal_result.reason or proposal_result.status or "empty result")
        return abstain("no_plane_region", f"proposal source abstained: {reason}")

    candidates = _proposal_mask_candidates(
        proposal_result, usable_mask, str(frame), diagnostics)
    if not candidates:
        return ()

    grey = _masked_grey(image, usable_mask)
    segments = detect_segments(enhance(grey))
    diagnostics["segments"] = int(len(segments))
    if len(segments) < MINIMUM_SEGMENTS:
        return abstain("no_plane_region", f"only {len(segments)} segments detected")
    weights = structural_weights(grey, segments)
    axis_matrix = np.asarray(
        axis_directions(calibration_result.yaw_degrees, calibration_result.pitch_degrees,
                        calibration_result.roll_degrees),
        dtype=float,
    )
    focal = float(calibration_result.focal_pixels)
    principal = calibration_result.principal_point or (
        calibration_result.width / 2.0, calibration_result.height / 2.0)
    principal = (float(principal[0]), float(principal[1]))
    height, width = usable_mask.shape
    weight_image = _weight_image(segments, weights, height, width)
    lattice._period_support_event("frame_structural_source", frame=frame,
        segments=segments, segment_weights=weights, calibration=calibration_result,
        grey=grey, usable=usable_mask, structural_weights=weight_image)
    return _measure_plane_candidates(
        grey, usable_mask, weight_image, candidates, axis_matrix, focal, principal,
        frame=frame, max_readings=max_readings,
        min_structural_share=min_structural_share,
        max_rectified_side=max_rectified_side, diagnostics=diagnostics,
    )


def plane_signature(
    image: np.ndarray,
    reading: PlaneReading,
    blocks_per_period: float,
    *,
    usable: np.ndarray | None = None,
    max_cells: int = MAX_SIGNATURE_CELLS,
    witness_sink=None,
    max_witness_memberships: int = 40_000_000,
    witness_budget: lattice.WitnessBudgetV1 | None = None,
) -> PlaneSignature:
    """Rectify one reading at one distance hypothesis into a cell signature.

    ``blocks_per_period`` must select exactly one retained
    :class:`DistanceAlternative`; its stored distance is authoritative for the
    signature and the graph homography. The cell grid is a
    square window centred on the reading's own region and capped at
    ``max_cells`` per side; each cell's seam-band and interior gradient energy
    come from a rectified patch that never exceeds :data:`MAX_RECTIFIED_SIDE`
    pixels per side.

    Proposal-region readings require the caller's explicit usable mask so
    their image-supported footprint cannot silently treat excluded pixels as
    valid.
    """
    if reading.rectification is None:  # pragma: no cover - the field is required
        raise ValueError("reading carries no rectification context")
    try:
        requested_b = float(blocks_per_period)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("blocks_per_period must be finite and positive") from error
    if not np.isfinite(requested_b) or requested_b <= 0.0:
        raise ValueError("blocks_per_period must be finite and positive")
    if reading.region_evidence is not None and usable is None:
        raise ValueError("proposal-region signatures require an explicit usable mask")
    matches = reading.matching_alternatives(requested_b)
    if len(matches) != 1:
        raise ValueError(
            "blocks_per_period must match exactly one retained reading alternative")
    alternative = matches[0]
    blocks_per_period = float(alternative.blocks_per_period)
    alternative_distance = float(alternative.distance_blocks)
    if not np.isfinite(alternative_distance) or alternative_distance <= 0.0:
        raise ValueError("matched reading alternative has an invalid distance")
    if usable is not None:
        usable = np.asarray(usable, dtype=bool)
        if usable.shape != np.asarray(image).shape[:2]:
            raise ValueError("usable mask does not match the image shape")

    rect = reading.rectification
    axis_matrix = np.asarray(rect.axis_matrix, dtype=float)
    try:
        pixel_step = lattice.rectified_pixel_step(reading.extent, reading.samples)
    except ValueError as error:
        raise ValueError("reading has no usable period/extent/sampling") from error
    world = reading.period_pixels * pixel_step
    if world <= 0.0 or not np.isfinite(world):
        raise ValueError("reading has no usable period/extent/sampling")
    # Keep the retained distance authoritative for both proposal and legacy
    # readings; historical/manual alternatives may use their original scale.
    distance = alternative_distance
    if distance <= 0.0 or not np.isfinite(distance):
        raise ValueError("blocks_per_period must be a positive finite number")

    centre_blocks = (rect.centre[0] * distance, rect.centre[1] * distance)
    half_blocks = reading.extent * distance
    if half_blocks <= 0.0 or not np.isfinite(half_blocks):
        raise ValueError("reading window has no block extent at this hypothesis")
    cells_side = max(1, int(np.ceil(2.0 * half_blocks)))
    origin_u = float(np.floor(centre_blocks[0] - half_blocks))
    origin_v = float(np.floor(centre_blocks[1] - half_blocks))
    if cells_side > int(max_cells):
        trim = (cells_side - int(max_cells)) // 2
        origin_u += trim
        origin_v += trim
        cells_side = int(max_cells)

    sub = max(1, min(SIGNATURE_PIXELS_PER_CELL, MAX_RECTIFIED_SIDE // max(cells_side, 1)))
    pixels = cells_side * sub
    unit_extent = (cells_side / 2.0) / distance
    unit_centre = ((origin_u + cells_side / 2.0) / distance,
                   (origin_v + cells_side / 2.0) / distance)

    rectification_witnesses = [] if witness_sink is not None else None
    if witness_sink is not None:
        budget = witness_budget or lattice.WitnessBudgetV1(
            max_cell_memberships=max_witness_memberships)
        rectifications = 3 if reading.region_evidence is not None else 2
        samples_planned = rectifications * pixels * pixels
        taps_planned = 4 * samples_planned
        memberships_planned = cells_side * cells_side * min(
            np.asarray(image).shape[0] * np.asarray(image).shape[1] + 1,
            16 * sub * sub)
        payload_planned = (samples_planned * 17 + taps_planned * 10
                           + memberships_planned * 8
                           + (cells_side * cells_side + rectifications) * 8)
        exhaustion = next(((stage, limit, planned) for stage, limit, planned in (
            ("aggregate_rectified_samples", budget.max_rectified_samples,
             samples_planned),
            ("aggregate_bilinear_taps", budget.max_bilinear_taps, taps_planned),
            ("aggregate_cell_memberships", budget.max_cell_memberships,
             memberships_planned),
            ("aggregate_payload_bytes", budget.max_payload_bytes, payload_planned),
        ) if planned > limit), None)
        if exhaustion is not None:
            stage, limit, planned = exhaustion
            witness_sink.append(lattice.WitnessRefusalV1(
                "refused", "budget_exhausted", stage, limit, planned, 0))
            rectification_witnesses = None
    def rectify(value):
        kwargs = dict(extent=unit_extent, samples=pixels, centre=unit_centre,
                      principal_point=rect.principal_point, side=reading.side)
        if rectification_witnesses is not None:
            kwargs["witness_sink"] = rectification_witnesses
        return lattice.rectify_plane(
            value, reading.axis, axis_matrix, rect.focal, **kwargs)
    patch = rectify(np.asarray(image))
    if patch is None or float(np.asarray(patch, dtype=float).sum()) <= 0.0:
        raise ValueError("reading could not be rectified at this hypothesis")
    coverage_image = (np.ones_like(np.asarray(image)[..., 0], dtype=np.float32)
                      if usable is None
                      else np.asarray(usable, dtype=np.float32))
    coverage_patch = rectify(coverage_image)
    if coverage_patch is None:
        raise ValueError("reading coverage could not be rectified at this hypothesis")

    grey = np.asarray(patch, dtype=np.float32)
    if grey.ndim == 3:
        grey = grey.mean(axis=2)
    gradient_y, gradient_x = np.gradient(grey)
    energy = np.hypot(gradient_x, gradient_y)
    covered = np.asarray(coverage_patch) > 0.5

    cells = np.zeros((cells_side, cells_side), dtype=np.float32)
    interior = np.zeros((cells_side, cells_side), dtype=np.float32)
    coverage = np.zeros((cells_side, cells_side), dtype=bool)
    region_footprint = None
    band = max(1, sub // 4)
    for row in range(cells_side):
        r0, r1 = row * sub, (row + 1) * sub
        for column in range(cells_side):
            c0, c1 = column * sub, (column + 1) * sub
            block = energy[r0:r1, c0:c1]
            coverage[row, column] = bool(np.count_nonzero(covered[r0:r1, c0:c1]) * 2
                                         >= block.size)
            cells[row, column] = float(block.sum())
            if sub > 2 * band:
                interior[row, column] = float(block[band:-band, band:-band].sum())
            else:
                interior[row, column] = float(block.sum())

    if reading.region_evidence is not None:
        region_mask = _decode_region_evidence(
            reading.region_evidence, reading.axis, reading.polygon)
        if region_mask.shape != np.asarray(image).shape[:2]:
            raise ValueError("reading region evidence does not match image dimensions")
        if usable is not None:
            region_mask &= usable
        region_patch = rectify(region_mask.astype(np.float32))
        if region_patch is None:
            raise ValueError("reading region mask could not be rectified at this hypothesis")
        inside = np.asarray(region_patch, dtype=float) > 0.5
        valid = np.asarray(coverage_patch, dtype=float) > 0.5
        if inside.shape != (pixels, pixels) or valid.shape != (pixels, pixels):
            raise ValueError("reading region coverage has unexpected rectified dimensions")
        inside_counts = []
        valid_counts = []
        for row in range(cells_side):
            r0, r1 = row * sub, (row + 1) * sub
            for column in range(cells_side):
                c0, c1 = column * sub, (column + 1) * sub
                inside_count = int(np.count_nonzero(inside[r0:r1, c0:c1]))
                valid_count = int(np.count_nonzero(valid[r0:r1, c0:c1]))
                if inside_count > valid_count:
                    raise ValueError("region samples exceed valid samples")
                inside_counts.append(inside_count)
                valid_counts.append(valid_count)
        region_footprint = PlaneRegionFootprint(
            source_region_id=reading.region_evidence.source_region_id,
            blocks_per_period=float(blocks_per_period),
            origin_uv=(origin_u, origin_v),
            grid_shape=(cells_side, cells_side),
            subsample_factor=sub,
            inside_region_sample_counts=bytes(inside_counts),
            valid_sample_counts=bytes(valid_counts),
            total_samples_per_cell=sub * sub,
        )

    result = PlaneSignature(
        reading=reading,
        blocks_per_period=float(blocks_per_period),
        cells=cells,
        interior=interior,
        coverage=coverage,
        origin_uv=(origin_u, origin_v),
        region_footprint=region_footprint,
    )
    if rectification_witnesses is not None:
        rgb_witness, coverage_witness = rectification_witnesses[:2]
        pointers, source_indices = [0], []
        rectangles, interior_rectangles = [], []
        for row in range(cells_side):
            r0, r1 = row * sub, (row + 1) * sub
            for column in range(cells_side):
                c0, c1 = column * sub, (column + 1) * sub
                rectangles.append((r0, r1, c0, c1))
                interior_rectangles.append(
                    (r0 + band, r1 - band, c0 + band, c1 - band)
                    if sub > 2 * band else (r0, r1, c0, c1))
                dependencies = set()
                for rr in range(r0, r1):
                    y_rows = ((0, 1) if rr == 0 else
                              ((pixels - 2, pixels - 1) if rr == pixels - 1
                               else (rr - 1, rr + 1)))
                    for cc in range(c0, c1):
                        x_columns = ((0, 1) if cc == 0 else
                                     ((pixels - 2, pixels - 1)
                                      if cc == pixels - 1 else (cc - 1, cc + 1)))
                        for sample_row, sample_column in (
                                {(yr, cc) for yr in y_rows}
                                | {(rr, xc) for xc in x_columns}):
                            sample = sample_row * pixels + sample_column
                            start = int(rgb_witness.tap_ptr[sample])
                            end = int(rgb_witness.tap_ptr[sample + 1])
                            dependencies.update(int(value) for value in
                                                rgb_witness.tap_source_indices[start:end])
                ordered = sorted(dependencies)
                if len(source_indices) + len(ordered) > budget.max_cell_memberships:
                    raise RuntimeError("witness membership preflight was inconsistent")
                source_indices.extend(ordered)
                pointers.append(len(source_indices))
        def digest(array):
            return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
        pointer_array = np.asarray(pointers, dtype=np.int64)
        pointer_array.flags.writeable = False
        member_array = np.asarray(source_indices, dtype=np.int64)
        member_array.flags.writeable = False
        witness_sink.append(SignatureSamplingWitnessV1(
            "mc-photogrammetry-signature-sampling-witness-v1", rgb_witness,
            coverage_witness,
            rectification_witnesses[2] if len(rectification_witnesses) == 3 else None,
            digest(grey), digest(gradient_y), digest(gradient_x), digest(energy),
            digest(coverage_patch), 1, sub, band, tuple(rectangles),
            tuple(interior_rectangles), pointer_array, member_array))
    capture = lattice._period_support_state()
    if capture is not None and capture.context.get("signature_target"):
        _period_support_signature_witness(result=result, grey=grey, gradient_y=gradient_y,
            gradient_x=gradient_x, energy=energy, coverage_patch=coverage_patch,
            pixels=pixels, sub=sub, band=band)
    return result


def _period_support_signature_witness(*, result, grey, gradient_y, gradient_x,
                                      energy, coverage_patch, pixels, sub, band):
    def build(capture):
        capture.memory.reserve(2048 * pixels * pixels + 1024**2,
                               "signature_membership_sets_lists_and_buffers")
        witnesses = capture.rectifications
        expected = 3 if result.reading.region_evidence is not None else 2
        if len(witnesses) != expected:
            raise lattice._PeriodSupportIncomplete("missing_dependency", "signature_rectifications")
        rgb_witness, coverage_witness = witnesses[:2]
        pointers, members, rectangles, interiors = [0], [], [], []
        rows, columns = result.cells.shape
        for row in range(rows):
            r0, r1 = row * sub, (row + 1) * sub
            for column in range(columns):
                c0, c1 = column * sub, (column + 1) * sub
                rectangles.append((r0, r1, c0, c1))
                interiors.append((r0 + band, r1 - band, c0 + band, c1 - band)
                                 if sub > 2 * band else (r0, r1, c0, c1))
                dependencies = set()
                for rr in range(r0, r1):
                    ys = (0, 1) if rr == 0 else ((pixels - 2, pixels - 1) if rr == pixels - 1 else (rr - 1, rr + 1))
                    for cc in range(c0, c1):
                        xs = (0, 1) if cc == 0 else ((pixels - 2, pixels - 1) if cc == pixels - 1 else (cc - 1, cc + 1))
                        for yr, xc in ({(yr, cc) for yr in ys} | {(rr, xc) for xc in xs}):
                            sample = yr * pixels + xc
                            start, stop = int(rgb_witness.tap_ptr[sample]), int(rgb_witness.tap_ptr[sample + 1])
                            dependencies.update(int(value) for value in rgb_witness.tap_source_indices[start:stop])
                if len(members) + len(dependencies) > 1048576:
                    raise lattice._PeriodSupportIncomplete("cap_exhausted", "signature_source_memberships")
                members.extend(sorted(dependencies)); pointers.append(len(members))
        pointer_array = np.frombuffer(np.asarray(pointers, dtype=np.int64).tobytes(), dtype=np.int64)
        member_array = np.frombuffer(np.asarray(members, dtype=np.int64).tobytes(), dtype=np.int64)
        def digest(array):
            return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
        witness = SignatureSamplingWitnessV1(
            "mc-photogrammetry-signature-sampling-witness-v1", rgb_witness,
            coverage_witness, witnesses[2] if expected == 3 else None,
            digest(grey), digest(gradient_y), digest(gradient_x), digest(energy),
            digest(coverage_patch), 1, sub, band, tuple(rectangles), tuple(interiors),
            pointer_array, member_array)
        lattice._period_support_event("signature_sampling", metadata=witness.to_record(),
            signature=result, grayscale=grey, gradient_y=gradient_y, gradient_x=gradient_x,
            energy=energy, coverage_patch=coverage_patch, cell_source_ptr=pointer_array,
            cell_source_indices=member_array)
    lattice._period_support_private(build)


def reading_order_key(reading: "PlaneReading") -> tuple:
    """Content-only key that orders a frame's readings independently of input order.

    :func:`~mc_photogrammetry.reconstruction.plane_graph.solve_component` and
    :func:`~mc_photogrammetry.reconstruction.plane_graph.match_plane_graph`
    address a reading by ``(frame, index)``, so the index must not depend on the
    order the caller happened to pass readings in. A source-region id breaks ties
    between otherwise numerically identical proposal readings; absent provenance
    retains the legacy empty tie-break.
    """
    return (
        reading.axis, reading.side, reading.period_pixels, reading.phase_pixels,
        reading.extent, reading.samples, reading.polygon or (),
        reading.structural_share, reading.confidence,
        tuple((alternative.blocks_per_period, alternative.distance_blocks,
               alternative.support) for alternative in reading.alternatives),
        (reading.region_evidence.source_region_id
         if reading.region_evidence is not None else ""),
    )


@dataclass(frozen=True, slots=True)
class WarpSignatureWitnessV1:
    schema: str
    homography: np.ndarray
    source_shape: tuple[int, int]
    target_shape: tuple[int, int]
    mapped_u: np.ndarray
    mapped_v: np.ndarray
    geometric_valid: np.ndarray
    source_rows: np.ndarray
    source_columns: np.ndarray
    witness_sha256: str = ""

    def __post_init__(self) -> None:
        arrays = (self.homography, self.mapped_u, self.mapped_v,
                  self.geometric_valid, self.source_rows, self.source_columns)
        if (self.schema != "mc-photogrammetry-warp-signature-witness-v1"
                or any(not isinstance(value, np.ndarray) or value.flags.writeable
                       for value in arrays)
                or self.homography.shape != (3, 3)
                or self.mapped_u.shape != self.target_shape
                or self.mapped_v.shape != self.target_shape
                or self.geometric_valid.shape != self.target_shape
                or self.source_rows.shape != self.target_shape
                or self.source_columns.shape != self.target_shape
                or self.geometric_valid.dtype != np.bool_
                or self.source_rows.dtype != np.int64
                or self.source_columns.dtype != np.int64):
            raise ValueError("warp signature witness is malformed or mutable")
        digest = hashlib.sha256(json.dumps(
            self.to_record(include_digest=False), sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if self.witness_sha256 not in ("", digest):
            raise ValueError("warp signature witness digest mismatch")
        object.__setattr__(self, "witness_sha256", digest)

    def to_record(self, *, include_digest=True):
        def row(value):
            return {"dtype": value.dtype.str, "shape": list(value.shape),
                    "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}
        result = {"schema": self.schema, "homography": row(self.homography),
                  "source_shape": list(self.source_shape),
                  "target_shape": list(self.target_shape),
                  "mapped_u": row(self.mapped_u), "mapped_v": row(self.mapped_v),
                  "geometric_valid": row(self.geometric_valid),
                  "source_rows": row(self.source_rows),
                  "source_columns": row(self.source_columns),
                  "rounding": "np.rint-clip-nearest-v1"}
        if include_digest:
            result["witness_sha256"] = self.witness_sha256
        return result


def warp_signature(
    signature: PlaneSignature,
    homography: np.ndarray,
    target_shape: tuple[int, int],
    *, witness_sink=None, witness_budget: lattice.WitnessBudgetV1 | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a signature's cell map by a homography, nearest-neighbour.

    ``homography`` maps *target* cell indices ``(column, row)`` to *source* cell
    indices; the caller bakes the two grids' ``origin_uv`` into it. Returns the
    warped ``cells`` map and its coverage. Resampling is at cell resolution only,
    exactly as the DS-28 directive requires — no sub-cell interpolation can
    invent lattice detail.
    """
    rows, columns = int(target_shape[0]), int(target_shape[1])
    if rows <= 0 or columns <= 0:
        raise ValueError("target_shape must be positive")
    source_rows, source_columns = signature.cells.shape
    grid_rows, grid_columns = np.mgrid[0:rows, 0:columns]
    points = np.stack((
        grid_columns.astype(float),
        grid_rows.astype(float),
        np.ones((rows, columns), dtype=float),
    ), axis=0)
    mapped = np.asarray(homography, dtype=float) @ points.reshape(3, -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        source_u = mapped[0] / mapped[2]
        source_v = mapped[1] / mapped[2]
    valid = (np.isfinite(source_u) & np.isfinite(source_v)
             & (source_u >= -0.5) & (source_u <= source_columns - 0.5)
             & (source_v >= -0.5) & (source_v <= source_rows - 0.5))
    warped = np.zeros((rows, columns), dtype=np.float32)
    coverage = np.zeros((rows, columns), dtype=bool)
    if valid.any():
        source_columns_index = np.clip(np.rint(source_u[valid]).astype(int), 0, source_columns - 1)
        source_rows_index = np.clip(np.rint(source_v[valid]).astype(int), 0, source_rows - 1)
        flat = np.flatnonzero(valid.reshape(-1))
        warped.reshape(-1)[flat] = signature.cells[source_rows_index, source_columns_index]
        coverage.reshape(-1)[flat] = signature.coverage[source_rows_index, source_columns_index]
    if witness_sink is not None:
        budget = witness_budget or lattice.WitnessBudgetV1()
        entries = rows * columns
        payload = entries * (8 + 8 + 1 + 8 + 8) + 9 * 8
        if entries > budget.max_pair_entries or payload > budget.max_payload_bytes:
            stage, limit, planned = (("warp_entries", budget.max_pair_entries, entries)
                                     if entries > budget.max_pair_entries else
                                     ("warp_payload_bytes", budget.max_payload_bytes, payload))
            witness_sink.append(lattice.WitnessRefusalV1(
                "refused", "budget_exhausted", stage, limit, planned, 0))
        else:
            source_row_grid = np.full((rows, columns), -1, dtype=np.int64)
            source_column_grid = np.full((rows, columns), -1, dtype=np.int64)
            if valid.any():
                source_row_grid.reshape(-1)[flat] = source_rows_index
                source_column_grid.reshape(-1)[flat] = source_columns_index
            frozen = []
            for value in (np.asarray(homography, dtype=np.float64).copy(),
                          source_u.reshape(rows, columns).astype(np.float64),
                          source_v.reshape(rows, columns).astype(np.float64),
                          valid.reshape(rows, columns).copy(), source_row_grid,
                          source_column_grid):
                value.flags.writeable = False; frozen.append(value)
            witness_sink.append(WarpSignatureWitnessV1(
                "mc-photogrammetry-warp-signature-witness-v1", frozen[0],
                (source_rows, source_columns), (rows, columns), *frozen[1:]))
    capture = lattice._period_support_state()
    if capture is not None and capture.context.get("match_target"):
        def observe_warp(state):
            entries = rows * columns
            if entries > 1048576 or entries * 33 + 72 > 32 * 1024**2:
                raise lattice._PeriodSupportIncomplete("cap_exhausted", "warp_mapping")
            state.memory.reserve(256 * entries + 1024**2, "warp_mapping_buffers")
            source_row_grid = np.full((rows, columns), -1, dtype=np.int64)
            source_column_grid = np.full((rows, columns), -1, dtype=np.int64)
            if valid.any():
                source_row_grid.reshape(-1)[flat] = source_rows_index
                source_column_grid.reshape(-1)[flat] = source_columns_index
            def detached(value):
                return np.frombuffer(value.tobytes(order="C"), dtype=value.dtype).reshape(value.shape)
            witness = WarpSignatureWitnessV1(
                "mc-photogrammetry-warp-signature-witness-v1",
                detached(np.asarray(homography, dtype=np.float64)),
                (source_rows, source_columns), (rows, columns),
                detached(source_u.reshape(rows, columns).astype(np.float64)),
                detached(source_v.reshape(rows, columns).astype(np.float64)),
                detached(valid.reshape(rows, columns)), detached(source_row_grid),
                detached(source_column_grid))
            state.warp = witness
            lattice._period_support_event("matcher_warp", witness=witness,
                warped_cells=warped, warped_coverage=coverage)
        lattice._period_support_private(observe_warp)
    return warped, coverage


# ---------------------------------------------------------------------------
# M4 V2-B P0 split-phase producer (frozen API and schema).
#
# Implements only the frozen public names of
# docs/reports/M4-V2B-P0-split-phase-producer-contract-2026-09-14.md.  The
# production function is a pure in-memory transform: it performs no IO, never
# imports plane_residue, never creates a PeriodicResidueCandidates, and never
# emits a boundary/interior energy, a hypothesis, a translation, or a tau.
# ---------------------------------------------------------------------------

SPLIT_PHASE_PRODUCER_VERSION = b"v2b-p0-split-phase-producer-1"
"""Frozen producer version bytes (never a string)."""

SPLIT_SELECTOR_VERSION = b"v2b-p0-selector-1"
"""Frozen selector version bytes (never a string)."""

SPLIT_SEAM_PROFILE_PAYLOAD_VERSION = b"v2b-p0-seam-profile-1"
"""Frozen payload-framing version bytes prefixed in every ``payload_sha256``."""

SPLIT_PHASE_PARTITION_VERSION = "v2b-p0-partition-1"
SPLIT_PHASE_SELECTOR_SALT = "v2b-p0-split-selector-salt-1"
SPLIT_PHASE_REFUSAL_DETAIL_MAX_BYTES = 4096

SPLIT_PHASE_REFUSAL_REASONS: tuple[str, ...] = (
    "split_not_reconstructible", "split_leakage", "split_replay_mismatch",
    "legacy_full_patch_nonpromotable", "selector_domain_invalid",
    "selector_min_lanes_unmet", "selector_role_invalid",
    "guard_footprint_uncoverable", "axis_lane_overlap", "empty_lane",
    "lane_plan_mismatch", "nonfinite_patch", "negative_weight",
    "noncanonical_dyadic", "g_phase_reconstruction_refused",
    "seam_profile_not_reconstructible", "seam_profile_payload_invalid",
    "refusal_detail_too_large", "partition_config_not_builtin",
    "work_over_cap", "budget_stage_invalid", "budget_exhausted",
    "no_feasible_calibration", "p0_patch_unavailable", "p0_patch_mismatch",
)
"""Closed typed-refusal enum; no other string is ever a refusal reason."""


def SPLIT_PHASE_CANONICAL_JSON(value) -> bytes:
    """Frozen canonical JSON encoding used for every split-phase digest."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


_SPLIT_PHASE_PATCH_DOMAIN = b"mc-photogrammetry-split-patch-v1"
_SPLIT_PHASE_GEN_UNAVAILABLE = {
    None, "insufficient_period", "empty_profile", "nonfinite_profile",
    "zero_resultant", "invalid_profile", "invalid_period",
}
_SPLIT_PHASE_SAMPLE_ENCODINGS = ("float64-le-base64", "float64-le-base64",
                                 "int64-le-base64")


def _split_hex64(value) -> bool:
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _split_digest(payload: dict, digest_field: str) -> str:
    material = {key: value for key, value in payload.items()
                if key != digest_field}
    return hashlib.sha256(SPLIT_PHASE_CANONICAL_JSON(material)).hexdigest()


def _split_detail(value):
    """Return the canonical JSON-safe form of a refusal detail, or raise."""
    if value is None or type(value) is bool or type(value) is int or type(value) is str:
        return value
    if type(value) is float:
        if not np.isfinite(value):
            raise ValueError("split_not_reconstructible")
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_split_detail(item) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("split_not_reconstructible")
            result[key] = _split_detail(item)
        return result
    raise ValueError("split_not_reconstructible")


def _split_reduced_dyadic(value, *, allow_zero: bool, allow_one: bool,
                          require_positive: bool = False) -> tuple[int, int]:
    """Validate and canonicalize a reduced dyadic ``(numerator, denominator)``."""
    if (not isinstance(value, tuple) or len(value) != 2
            or any(type(part) is not int for part in value)):
        raise ValueError("noncanonical_dyadic")
    numerator, denominator = value
    if denominator <= 0 or (denominator & (denominator - 1)) != 0:
        raise ValueError("noncanonical_dyadic")
    if numerator == 0:
        if denominator != 1 or not allow_zero or require_positive:
            raise ValueError("noncanonical_dyadic")
        return (0, 1)
    if numerator < 0 or numerator > denominator:
        raise ValueError("noncanonical_dyadic")
    if numerator == denominator:
        if not allow_one:
            raise ValueError("noncanonical_dyadic")
        return (1, 1)
    if numerator % 2 == 0:
        raise ValueError("noncanonical_dyadic")
    return (numerator, denominator)


def _split_finite_float(value) -> float:
    if type(value) not in (int, float) or not np.isfinite(value):
        raise ValueError("split_not_reconstructible")
    return float(value)


def _split_pair(value) -> tuple[float, float]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError("split_not_reconstructible")
    return (_split_finite_float(value[0]), _split_finite_float(value[1]))


@dataclass(frozen=True, slots=True)
class SplitPhaseGeometry:
    """Frozen rectified-patch geometry: exactly five ordered fields."""

    centre: tuple[float, float]
    extent: float
    samples: int
    rectified_pixel_step: float
    period_pixels: float

    def __post_init__(self) -> None:
        centre = _split_pair(self.centre)
        extent = _split_finite_float(self.extent)
        step = _split_finite_float(self.rectified_pixel_step)
        period = _split_finite_float(self.period_pixels)
        if (type(self.samples) is not int or self.samples < 2
                or extent <= 0.0 or period <= 0.0 or step <= 0.0):
            raise ValueError("split_not_reconstructible")
        expected = float(lattice.rectified_pixel_step(extent, self.samples))
        if step != expected:
            raise ValueError("split_not_reconstructible")
        object.__setattr__(self, "centre", centre)
        object.__setattr__(self, "extent", extent)
        object.__setattr__(self, "rectified_pixel_step", step)
        object.__setattr__(self, "period_pixels", period)

    def to_record(self) -> dict:
        """Exactly the five ordered JSON-safe geometry fields."""
        return {
            "centre": [self.centre[0], self.centre[1]],
            "extent": self.extent,
            "samples": self.samples,
            "rectified_pixel_step": self.rectified_pixel_step,
            "period_pixels": self.period_pixels,
        }


def _split_geometry_sha256(geometry: SplitPhaseGeometry) -> str:
    return hashlib.sha256(
        SPLIT_PHASE_CANONICAL_JSON(geometry.to_record())).hexdigest()


@dataclass(frozen=True, slots=True)
class SplitPhasePartitionConfig:
    """Frozen partition/selector configuration (seven ordered fields)."""

    partition_version: str
    selector_version: bytes
    selector_salt: str
    min_role_lanes: int
    profile_stencil_radius_samples: int
    seam_stencil_radius_samples: int
    guard_lanes: int

    def __post_init__(self) -> None:
        if (type(self.partition_version) is not str or not self.partition_version
                or type(self.selector_version) is not bytes or not self.selector_version
                or type(self.selector_salt) is not str or not self.selector_salt):
            raise ValueError("split_not_reconstructible")
        for name in ("min_role_lanes", "profile_stencil_radius_samples",
                     "seam_stencil_radius_samples", "guard_lanes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError("split_not_reconstructible")
        if self.min_role_lanes < 1:
            raise ValueError("split_not_reconstructible")
        if self.guard_lanes < max(self.profile_stencil_radius_samples,
                                  self.seam_stencil_radius_samples):
            raise ValueError("guard_footprint_uncoverable")

    def to_record(self) -> dict:
        return {
            "partition_version": self.partition_version,
            "selector_version_hex": self.selector_version.hex(),
            "selector_salt": self.selector_salt,
            "min_role_lanes": self.min_role_lanes,
            "profile_stencil_radius_samples": self.profile_stencil_radius_samples,
            "seam_stencil_radius_samples": self.seam_stencil_radius_samples,
            "guard_lanes": self.guard_lanes,
        }


def partition_config_sha256(partition_config: SplitPhasePartitionConfig) -> str:
    """Digest over the seven-field partition-config record."""
    if not isinstance(partition_config, SplitPhasePartitionConfig):
        raise ValueError("partition_config_not_builtin")
    return hashlib.sha256(
        SPLIT_PHASE_CANONICAL_JSON(partition_config.to_record())).hexdigest()


SPLIT_PHASE_PARTITION_CONFIG = SplitPhasePartitionConfig(
    partition_version=SPLIT_PHASE_PARTITION_VERSION,
    selector_version=SPLIT_SELECTOR_VERSION,
    selector_salt=SPLIT_PHASE_SELECTOR_SALT,
    min_role_lanes=1,
    profile_stencil_radius_samples=0,
    seam_stencil_radius_samples=0,
    guard_lanes=0,
)
"""The only production partition config; never a caller value."""

SPLIT_PHASE_PARTITION_CONFIG_SHA256 = partition_config_sha256(
    SPLIT_PHASE_PARTITION_CONFIG)

_SPLIT_BUDGET_FIELDS: tuple[str, ...] = (
    "max_lanes_per_axis", "max_patch_samples", "max_inline_patch_samples",
    "max_g_samples_per_axis", "max_d_samples_per_axis",
    "max_guard_samples_per_axis", "max_profile_terms", "max_d_profile_terms",
    "max_payload_bytes", "max_source_records", "max_replay_checks",
)
_SPLIT_WORK_FIELDS: tuple[str, ...] = (
    "lanes_per_axis", "patch_samples", "inline_patch_samples",
    "g_samples_per_axis", "d_samples_per_axis", "guard_samples_per_axis",
    "profile_terms", "d_profile_terms", "payload_bytes", "source_records",
    "replay_checks",
)
_SPLIT_WORK_TO_CAP = dict(zip(_SPLIT_WORK_FIELDS, _SPLIT_BUDGET_FIELDS))


@dataclass(frozen=True, slots=True)
class SplitPhaseProducerBudget:
    """Eleven explicit positive exact-integer caps plus their self digest."""

    max_lanes_per_axis: int
    max_patch_samples: int
    max_inline_patch_samples: int
    max_g_samples_per_axis: int
    max_d_samples_per_axis: int
    max_guard_samples_per_axis: int
    max_profile_terms: int
    max_d_profile_terms: int
    max_payload_bytes: int
    max_source_records: int
    max_replay_checks: int
    budget_sha256: str

    def __post_init__(self) -> None:
        for name in _SPLIT_BUDGET_FIELDS:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError("split_not_reconstructible")
        if (not _split_hex64(self.budget_sha256)
                or _split_digest(self.to_record(), "budget_sha256")
                != self.budget_sha256):
            raise ValueError("split_not_reconstructible")

    def to_record(self) -> dict:
        record = {name: getattr(self, name) for name in _SPLIT_BUDGET_FIELDS}
        record["budget_sha256"] = self.budget_sha256
        return record


def _make_budget(caps: dict) -> SplitPhaseProducerBudget:
    payload = {name: int(caps[name]) for name in _SPLIT_BUDGET_FIELDS}
    return SplitPhaseProducerBudget(
        **payload, budget_sha256=_split_digest(payload, "budget_sha256"))


@dataclass(frozen=True, slots=True)
class SplitPhaseProducerWork:
    """Eleven nonnegative planned/complete counters plus their self digest."""

    lanes_per_axis: int
    patch_samples: int
    inline_patch_samples: int
    g_samples_per_axis: int
    d_samples_per_axis: int
    guard_samples_per_axis: int
    profile_terms: int
    d_profile_terms: int
    payload_bytes: int
    source_records: int
    replay_checks: int
    work_sha256: str

    def __post_init__(self) -> None:
        for name in _SPLIT_WORK_FIELDS:
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError("split_not_reconstructible")
        if (not _split_hex64(self.work_sha256)
                or _split_digest(self.to_record(), "work_sha256")
                != self.work_sha256):
            raise ValueError("split_not_reconstructible")

    def to_record(self) -> dict:
        record = {name: getattr(self, name) for name in _SPLIT_WORK_FIELDS}
        record["work_sha256"] = self.work_sha256
        return record


def _make_work(counters: dict) -> SplitPhaseProducerWork:
    payload = {name: int(counters.get(name, 0)) for name in _SPLIT_WORK_FIELDS}
    return SplitPhaseProducerWork(
        **payload, work_sha256=_split_digest(payload, "work_sha256"))


@dataclass(frozen=True, slots=True)
class SplitPhaseRefusal:
    """Frozen bounded typed refusal (four ordered fields)."""

    reason: str
    detail: object
    detail_sha256: str | None
    refusal_sha256: str

    def __post_init__(self) -> None:
        if type(self.reason) is not str or self.reason not in SPLIT_PHASE_REFUSAL_REASONS:
            raise ValueError("split_not_reconstructible")
        normalized = _split_detail(self.detail)
        encoded = SPLIT_PHASE_CANONICAL_JSON(normalized)
        if len(encoded) > SPLIT_PHASE_REFUSAL_DETAIL_MAX_BYTES:
            raise ValueError("refusal_detail_too_large")
        expected = None if normalized is None else hashlib.sha256(encoded).hexdigest()
        if self.detail_sha256 != expected:
            raise ValueError("split_not_reconstructible")
        payload = {"reason": self.reason, "detail_sha256": self.detail_sha256}
        if (not _split_hex64(self.refusal_sha256)
                or _split_digest(payload, "refusal_sha256") != self.refusal_sha256):
            raise ValueError("split_not_reconstructible")
        object.__setattr__(self, "detail", normalized)

    def to_record(self) -> dict:
        return {
            "reason": self.reason,
            "detail": self.detail,
            "detail_sha256": self.detail_sha256,
            "refusal_sha256": self.refusal_sha256,
        }


def _make_refusal(reason: str, detail=None, cap=None) -> SplitPhaseRefusal:
    if reason not in SPLIT_PHASE_REFUSAL_REASONS:
        raise ValueError("split_not_reconstructible")
    normalized = _split_detail(detail)
    encoded = SPLIT_PHASE_CANONICAL_JSON(normalized)
    if len(encoded) > SPLIT_PHASE_REFUSAL_DETAIL_MAX_BYTES:
        normalized = {"limit": SPLIT_PHASE_REFUSAL_DETAIL_MAX_BYTES,
                      "observed": len(encoded), "cap": cap}
        reason = "refusal_detail_too_large"
        encoded = SPLIT_PHASE_CANONICAL_JSON(normalized)
    detail_sha = None if normalized is None else hashlib.sha256(encoded).hexdigest()
    payload = {"reason": reason, "detail_sha256": detail_sha}
    return SplitPhaseRefusal(reason, normalized, detail_sha,
                             _split_digest(payload, "refusal_sha256"))


@dataclass(frozen=True, slots=True)
class SplitPhaseLaneRole:
    """Frozen pure selector result (shape (i) role or shape (ii) refusal)."""

    role: str | None
    orientation: int | None
    refusal: str | None
    role_sha256: str

    def __post_init__(self) -> None:
        assigned = (self.role in ("G", "D", "guard")
                    and type(self.orientation) is int
                    and self.orientation in (0, 1) and self.refusal is None)
        refused = (self.role is None and self.orientation is None
                   and self.refusal == "selector_min_lanes_unmet")
        if not (assigned or refused):
            raise ValueError("selector_role_invalid")
        if (not _split_hex64(self.role_sha256)
                or _split_digest(self.to_record(), "role_sha256") != self.role_sha256):
            raise ValueError("split_not_reconstructible")

    def to_record(self) -> dict:
        return {
            "role": self.role,
            "orientation": self.orientation,
            "refusal": self.refusal,
            "role_sha256": self.role_sha256,
        }


def _make_lane_role(role, orientation, refusal) -> SplitPhaseLaneRole:
    payload = {"role": role, "orientation": orientation, "refusal": refusal}
    return SplitPhaseLaneRole(
        role=role, orientation=orientation, refusal=refusal,
        role_sha256=_split_digest(payload, "role_sha256"))


def _selector_domain(axis_label, lane_index, lane_count, orthogonal_extent,
                     partition_config) -> None:
    if not isinstance(partition_config, SplitPhasePartitionConfig):
        raise ValueError("selector_domain_invalid")
    if type(axis_label) is not str or axis_label not in ("u", "v"):
        raise ValueError("selector_domain_invalid")
    for value in (lane_index, lane_count, orthogonal_extent):
        if type(value) is not int:
            raise ValueError("selector_domain_invalid")
    if lane_count < 1 or orthogonal_extent < 1 or not 0 <= lane_index < lane_count:
        raise ValueError("selector_domain_invalid")


def _selector_orientation(axis_label, lane_count, orthogonal_extent,
                          partition_config) -> int:
    try:
        version = partition_config.selector_version.decode("ascii")
    except (AttributeError, UnicodeDecodeError) as error:
        raise ValueError("selector_domain_invalid") from error
    canonical = json.dumps({
        "axis": axis_label,
        "lane_count": int(lane_count),
        "orthogonal_extent": int(orthogonal_extent),
        "salt": partition_config.selector_salt,
        "selector_version": version,
    }, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).digest()[31] & 1


def _selector_blocks(orientation: int, lane_count: int, nG: int,
                     guard_lanes: int) -> tuple[tuple[int, int], tuple[int, int] | None]:
    if orientation == 0:
        g_range = (0, nG)
        guard_range = ((nG, nG + guard_lanes) if guard_lanes > 0 else None)
    else:
        g_range = (nG + guard_lanes, lane_count)
        guard_range = ((nG, nG + guard_lanes) if guard_lanes > 0 else None)
    return g_range, guard_range


def lane_role(axis_label, lane_index, lane_count, orthogonal_extent,
              partition_config) -> SplitPhaseLaneRole:
    """Frozen pure selector: domain check, then min-lane, then orientation."""
    _selector_domain(axis_label, lane_index, lane_count, orthogonal_extent,
                     partition_config)
    minimum = 2 * partition_config.min_role_lanes + partition_config.guard_lanes
    if lane_count < minimum:
        return _make_lane_role(None, None, "selector_min_lanes_unmet")
    orientation = _selector_orientation(
        axis_label, lane_count, orthogonal_extent, partition_config)
    remainder = lane_count - partition_config.guard_lanes
    nG = remainder // 2
    if orientation == 0:
        if lane_index < nG:
            role = "G"
        elif lane_index < nG + partition_config.guard_lanes:
            role = "guard"
        else:
            role = "D"
    else:
        if lane_index < nG:
            role = "D"
        elif lane_index < nG + partition_config.guard_lanes:
            role = "guard"
        else:
            role = "G"
    return _make_lane_role(role, orientation, None)


@dataclass(frozen=True, slots=True)
class SplitPhaseLanePlan:
    """Frozen deterministic lane plan (fourteen ordered fields)."""

    local_axis: str
    orientation: int
    lane_count: int
    orthogonal_extent: int
    guard_lanes: int
    nG: int
    nD: int
    g_start: int
    g_stop: int
    d_start: int
    d_stop: int
    guard_start: int | None
    guard_stop: int | None
    lane_plan_sha256: str

    def __post_init__(self) -> None:
        if (self.local_axis not in ("u", "v")
                or type(self.orientation) is not int or self.orientation not in (0, 1)):
            raise ValueError("lane_plan_mismatch")
        for name in ("lane_count", "orthogonal_extent", "guard_lanes", "nG", "nD",
                     "g_start", "g_stop", "d_start", "d_stop"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError("lane_plan_mismatch")
        if self.lane_count < 1 or self.orthogonal_extent < 1 or self.nG < 1 or self.nD < 1:
            raise ValueError("lane_plan_mismatch")
        if self.g_stop <= self.g_start or self.d_stop <= self.d_start:
            raise ValueError("lane_plan_mismatch")
        if self.nG != (self.lane_count - self.guard_lanes) // 2:
            raise ValueError("lane_plan_mismatch")
        if self.nD != self.lane_count - self.guard_lanes - self.nG:
            raise ValueError("lane_plan_mismatch")
        g_range, guard_range = _selector_blocks(
            self.orientation, self.lane_count, self.nG, self.guard_lanes)
        d_range = ((self.nG + self.guard_lanes, self.lane_count)
                   if self.orientation == 0 else (0, self.nG))
        if (self.g_start, self.g_stop) != g_range or (self.d_start, self.d_stop) != d_range:
            raise ValueError("lane_plan_mismatch")
        if guard_range is None:
            if self.guard_start is not None or self.guard_stop is not None:
                raise ValueError("lane_plan_mismatch")
        elif ((self.guard_start, self.guard_stop) != guard_range
              or self.guard_stop <= self.guard_start):
            raise ValueError("lane_plan_mismatch")
        blocks = [g_range, d_range]
        if guard_range is not None:
            blocks.append(guard_range)
        # The three contiguous blocks cover exactly [0, N) in some role order
        # (orientation 1 emits D before G), so compare by start, not by role.
        ordered_blocks = sorted(blocks)
        if (ordered_blocks[0][0] != 0 or ordered_blocks[-1][1] != self.lane_count
                or any(left[1] != right[0]
                       for left, right in zip(ordered_blocks, ordered_blocks[1:]))):
            raise ValueError("axis_lane_overlap")
        if (not _split_hex64(self.lane_plan_sha256)
                or _split_digest(self.to_record(), "lane_plan_sha256")
                != self.lane_plan_sha256):
            raise ValueError("split_not_reconstructible")

    def to_record(self) -> dict:
        return {
            "local_axis": self.local_axis,
            "orientation": self.orientation,
            "lane_count": self.lane_count,
            "orthogonal_extent": self.orthogonal_extent,
            "guard_lanes": self.guard_lanes,
            "nG": self.nG,
            "nD": self.nD,
            "g_start": self.g_start,
            "g_stop": self.g_stop,
            "d_start": self.d_start,
            "d_stop": self.d_stop,
            "guard_start": self.guard_start,
            "guard_stop": self.guard_stop,
            "lane_plan_sha256": self.lane_plan_sha256,
        }


def _make_lane_plan(local_axis, orientation, lane_count, orthogonal_extent,
                    guard_lanes, nG, nD, g_range, guard_range) -> SplitPhaseLanePlan:
    if orientation == 0:
        d_range = (nG + guard_lanes, lane_count)
    else:
        d_range = (0, nG)
    guard_start, guard_stop = guard_range if guard_range is not None else (None, None)
    payload = {
        "local_axis": local_axis,
        "orientation": orientation,
        "lane_count": lane_count,
        "orthogonal_extent": orthogonal_extent,
        "guard_lanes": guard_lanes,
        "nG": nG,
        "nD": nD,
        "g_start": g_range[0],
        "g_stop": g_range[1],
        "d_start": d_range[0],
        "d_stop": d_range[1],
        "guard_start": guard_start,
        "guard_stop": guard_stop,
    }
    return SplitPhaseLanePlan(
        **payload, lane_plan_sha256=_split_digest(payload, "lane_plan_sha256"))


def split_phase_lane_plan(axis_label, lane_count, orthogonal_extent,
                          partition_config) -> "SplitPhaseLanePlan | SplitPhaseRefusal":
    """Frozen pure lane plan; domain/min-lane failures are typed refusals."""
    if not isinstance(partition_config, SplitPhasePartitionConfig):
        return _make_refusal("selector_domain_invalid")
    if (type(axis_label) is not str or axis_label not in ("u", "v")
            or type(lane_count) is not int or type(orthogonal_extent) is not int
            or lane_count < 1 or orthogonal_extent < 1):
        return _make_refusal("selector_domain_invalid")
    minimum = 2 * partition_config.min_role_lanes + partition_config.guard_lanes
    if lane_count < minimum:
        return _make_refusal("selector_min_lanes_unmet")
    orientation = _selector_orientation(
        axis_label, lane_count, orthogonal_extent, partition_config)
    remainder = lane_count - partition_config.guard_lanes
    nG = remainder // 2
    nD = remainder - nG
    g_range, guard_range = _selector_blocks(
        orientation, lane_count, nG, partition_config.guard_lanes)
    return _make_lane_plan(axis_label, orientation, lane_count,
                           orthogonal_extent, partition_config.guard_lanes,
                           nG, nD, g_range, guard_range)


def _split_patch_sha256(rows: int, columns: int, byte_length: int, raw: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(_SPLIT_PHASE_PATCH_DOMAIN)
    digest.update(int(rows).to_bytes(8, "little", signed=False))
    digest.update(int(columns).to_bytes(8, "little", signed=False))
    digest.update(int(byte_length).to_bytes(8, "little", signed=False))
    digest.update(raw)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SplitPhasePatchRef:
    """Frozen patch reference (ten ordered fields)."""

    mode: str
    patch_sha256: str
    rows: int
    columns: int
    dtype: str
    byte_length: int
    finite_nonnegative: bool
    inline_samples_float64_le_base64: str | None
    inline_sample_count: int | None
    patch_ref_sha256: str

    def __post_init__(self) -> None:
        if (self.mode not in ("digest", "inline")
                or not _split_hex64(self.patch_sha256)
                or type(self.rows) is not int or type(self.columns) is not int
                or self.rows < 1 or self.columns < 1
                or self.dtype != "float64"
                or type(self.byte_length) is not int
                or self.byte_length != self.rows * self.columns * 8
                or self.finite_nonnegative is not True):
            raise ValueError("split_not_reconstructible")
        if self.mode == "inline":
            if (type(self.inline_samples_float64_le_base64) is not str
                    or type(self.inline_sample_count) is not int
                    or self.inline_sample_count != self.rows * self.columns):
                raise ValueError("split_not_reconstructible")
            try:
                raw = base64.b64decode(
                    self.inline_samples_float64_le_base64, validate=True)
            except (TypeError, ValueError) as error:
                raise ValueError("split_not_reconstructible") from error
            if base64.b64encode(raw).decode("ascii") != self.inline_samples_float64_le_base64:
                raise ValueError("split_not_reconstructible")
            if len(raw) != self.byte_length:
                raise ValueError("split_not_reconstructible")
            if _split_patch_sha256(self.rows, self.columns, self.byte_length,
                                   raw) != self.patch_sha256:
                raise ValueError("split_not_reconstructible")
        elif (self.inline_samples_float64_le_base64 is not None
              or self.inline_sample_count is not None):
            raise ValueError("split_not_reconstructible")
        if (not _split_hex64(self.patch_ref_sha256)
                or _split_digest(self.to_record(), "patch_ref_sha256")
                != self.patch_ref_sha256):
            raise ValueError("split_not_reconstructible")

    def to_record(self) -> dict:
        return {
            "mode": self.mode,
            "patch_sha256": self.patch_sha256,
            "rows": self.rows,
            "columns": self.columns,
            "dtype": self.dtype,
            "byte_length": self.byte_length,
            "finite_nonnegative": self.finite_nonnegative,
            "inline_samples_float64_le_base64": self.inline_samples_float64_le_base64,
            "inline_sample_count": self.inline_sample_count,
            "patch_ref_sha256": self.patch_ref_sha256,
        }


def _make_patch_ref(patch: np.ndarray, inline_limit: int) -> SplitPhasePatchRef:
    rows, columns = int(patch.shape[0]), int(patch.shape[1])
    raw = np.asarray(patch, dtype="<f8").tobytes(order="C")
    byte_length = len(raw)
    patch_sha = _split_patch_sha256(rows, columns, byte_length, raw)
    if rows * columns <= int(inline_limit):
        payload = {
            "mode": "inline",
            "patch_sha256": patch_sha,
            "rows": rows,
            "columns": columns,
            "dtype": "float64",
            "byte_length": byte_length,
            "finite_nonnegative": True,
            "inline_samples_float64_le_base64": base64.b64encode(raw).decode("ascii"),
            "inline_sample_count": rows * columns,
        }
    else:
        payload = {
            "mode": "digest",
            "patch_sha256": patch_sha,
            "rows": rows,
            "columns": columns,
            "dtype": "float64",
            "byte_length": byte_length,
            "finite_nonnegative": True,
            "inline_samples_float64_le_base64": None,
            "inline_sample_count": None,
        }
    return SplitPhasePatchRef(
        **payload, patch_ref_sha256=_split_digest(payload, "patch_ref_sha256"))


@dataclass(frozen=True, slots=True)
class SplitPhaseLaneSet:
    """Frozen lane-set record (eleven ordered fields)."""

    local_axis: str
    role: str
    lane_index_start: int
    lane_index_stop: int
    row_start: int | None
    row_stop: int | None
    column_start: int | None
    column_stop: int | None
    sample_count: int
    coordinate_sha256: str
    lane_set_sha256: str

    def __post_init__(self) -> None:
        if self.local_axis not in ("u", "v") or self.role not in ("G", "D", "guard"):
            raise ValueError("lane_plan_mismatch")
        for name in ("lane_index_start", "lane_index_stop", "sample_count"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError("lane_plan_mismatch")
        if self.lane_index_stop <= self.lane_index_start or self.sample_count < 1:
            raise ValueError("empty_lane")
        if self.local_axis == "u":
            if (type(self.row_start) is not int or type(self.row_stop) is not int
                    or self.row_start < 0 or self.row_stop <= self.row_start
                    or self.column_start is not None or self.column_stop is not None):
                raise ValueError("lane_plan_mismatch")
        elif (type(self.column_start) is not int or type(self.column_stop) is not int
              or self.column_start < 0 or self.column_stop <= self.column_start
              or self.row_start is not None or self.row_stop is not None):
            raise ValueError("lane_plan_mismatch")
        if not _split_hex64(self.coordinate_sha256):
            raise ValueError("lane_plan_mismatch")
        if (not _split_hex64(self.lane_set_sha256)
                or _split_digest(self.to_record(), "lane_set_sha256")
                != self.lane_set_sha256):
            raise ValueError("split_not_reconstructible")

    def to_record(self) -> dict:
        return {
            "local_axis": self.local_axis,
            "role": self.role,
            "lane_index_start": self.lane_index_start,
            "lane_index_stop": self.lane_index_stop,
            "row_start": self.row_start,
            "row_stop": self.row_stop,
            "column_start": self.column_start,
            "column_stop": self.column_stop,
            "sample_count": self.sample_count,
            "coordinate_sha256": self.coordinate_sha256,
            "lane_set_sha256": self.lane_set_sha256,
        }


def _make_lane_set(local_axis, role, lane_start, lane_stop, row_range,
                   column_range, sample_count, coordinate_sha256) -> SplitPhaseLaneSet:
    row_start, row_stop = row_range if row_range is not None else (None, None)
    column_start, column_stop = (column_range if column_range is not None
                                 else (None, None))
    payload = {
        "local_axis": local_axis,
        "role": role,
        "lane_index_start": lane_start,
        "lane_index_stop": lane_stop,
        "row_start": row_start,
        "row_stop": row_stop,
        "column_start": column_start,
        "column_stop": column_stop,
        "sample_count": sample_count,
        "coordinate_sha256": coordinate_sha256,
    }
    return SplitPhaseLaneSet(
        **payload, lane_set_sha256=_split_digest(payload, "lane_set_sha256"))


@dataclass(frozen=True, slots=True)
class SplitGenerationPhaseEvidence:
    """Frozen G-only generation evidence (twenty ordered fields)."""

    frame: str
    plane_index: int
    reading_axis: int
    period_pixels: float
    patch_centre_uv: tuple[float, float]
    extent: float
    samples: int
    source_region_id: str | None
    phase_uv_pixels: tuple[float | None, float | None]
    response_magnitude_uv: tuple[float | None, float | None]
    profile_mass_uv: tuple[float | None, float | None]
    concentration_uv: tuple[float | None, float | None]
    unavailable_reason_uv: tuple[str | None, str | None]
    g_profile_sha256: str
    partition_sha256: str
    G_u_sha256: str
    G_v_sha256: str
    G_u_sample_count: int
    G_v_sample_count: int
    generation_evidence_sha256: str

    def __post_init__(self) -> None:
        if type(self.frame) is not str or not self.frame:
            raise ValueError("g_phase_reconstruction_refused")
        if type(self.plane_index) is not int or self.plane_index < 0:
            raise ValueError("g_phase_reconstruction_refused")
        if type(self.reading_axis) is not int or self.reading_axis not in (0, 1, 2):
            raise ValueError("g_phase_reconstruction_refused")
        period = _split_finite_float(self.period_pixels)
        centre = _split_pair(self.patch_centre_uv)
        extent = _split_finite_float(self.extent)
        if type(self.samples) is not int or self.samples < 2 or period <= 0.0 or extent <= 0.0:
            raise ValueError("g_phase_reconstruction_refused")
        if self.source_region_id is not None and not _split_hex64(self.source_region_id):
            raise ValueError("g_phase_reconstruction_refused")
        for name in ("phase_uv_pixels", "response_magnitude_uv", "profile_mass_uv",
                     "concentration_uv", "unavailable_reason_uv"):
            value = getattr(self, name)
            if not isinstance(value, tuple) or len(value) != 2:
                raise ValueError("g_phase_reconstruction_refused")
        try:
            PeriodicEnergyPhaseEvidence(
                phase_uv_pixels=tuple(self.phase_uv_pixels),
                period_pixels=period,
                patch_centre_uv=centre,
                extent=extent,
                samples=self.samples,
                response_magnitude_uv=tuple(self.response_magnitude_uv),
                profile_mass_uv=tuple(self.profile_mass_uv),
                concentration_uv=tuple(self.concentration_uv),
                unavailable_reason_uv=tuple(self.unavailable_reason_uv),
                source_region_id=self.source_region_id,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("g_phase_reconstruction_refused") from error
        for name in ("G_u_sample_count", "G_v_sample_count"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError("g_phase_reconstruction_refused")
        for name in ("g_profile_sha256", "partition_sha256", "G_u_sha256",
                     "G_v_sha256"):
            if not _split_hex64(getattr(self, name)):
                raise ValueError("g_phase_reconstruction_refused")
        object.__setattr__(self, "period_pixels", period)
        object.__setattr__(self, "patch_centre_uv", centre)
        object.__setattr__(self, "extent", extent)
        if (not _split_hex64(self.generation_evidence_sha256)
                or _split_digest(self.to_record(), "generation_evidence_sha256")
                != self.generation_evidence_sha256):
            raise ValueError("split_not_reconstructible")

    def to_record(self) -> dict:
        return {
            "frame": self.frame,
            "plane_index": self.plane_index,
            "reading_axis": self.reading_axis,
            "period_pixels": self.period_pixels,
            "patch_centre_uv": [self.patch_centre_uv[0], self.patch_centre_uv[1]],
            "extent": self.extent,
            "samples": self.samples,
            "source_region_id": self.source_region_id,
            "phase_uv_pixels": list(self.phase_uv_pixels),
            "response_magnitude_uv": list(self.response_magnitude_uv),
            "profile_mass_uv": list(self.profile_mass_uv),
            "concentration_uv": list(self.concentration_uv),
            "unavailable_reason_uv": list(self.unavailable_reason_uv),
            "g_profile_sha256": self.g_profile_sha256,
            "partition_sha256": self.partition_sha256,
            "G_u_sha256": self.G_u_sha256,
            "G_v_sha256": self.G_v_sha256,
            "G_u_sample_count": self.G_u_sample_count,
            "G_v_sample_count": self.G_v_sample_count,
            "generation_evidence_sha256": self.generation_evidence_sha256,
        }


def _make_generation_evidence(record_fields: dict) -> SplitGenerationPhaseEvidence:
    return SplitGenerationPhaseEvidence(
        **record_fields,
        generation_evidence_sha256=_split_digest(
            record_fields, "generation_evidence_sha256"))


def _split_decode_array(encoded, dtype) -> tuple[bytes, np.ndarray]:
    if type(encoded) is not str:
        raise ValueError("seam_profile_payload_invalid")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (TypeError, ValueError) as error:
        raise ValueError("seam_profile_payload_invalid") from error
    if base64.b64encode(raw).decode("ascii") != encoded:
        raise ValueError("seam_profile_payload_invalid")
    if len(raw) % 8 != 0:
        raise ValueError("seam_profile_payload_invalid")
    return raw, np.frombuffer(raw, dtype=dtype)


def _split_payload_blob(samples_raw: bytes, weights_raw: bytes,
                        indices_raw: bytes) -> bytes:
    parts = []
    for raw in (samples_raw, weights_raw, indices_raw):
        parts.append(len(raw).to_bytes(8, "little", signed=False))
        parts.append(raw)
    return b"".join(parts)


def _split_payload_sha256(samples_raw: bytes, weights_raw: bytes,
                          indices_raw: bytes) -> str:
    return hashlib.sha256(
        SPLIT_SEAM_PROFILE_PAYLOAD_VERSION
        + _split_payload_blob(samples_raw, weights_raw, indices_raw)).hexdigest()


def _split_d_profile_sha256(local_axis, period_pixels, sample_count, weight_count,
                            index_count, coverage, payload_sha256) -> str:
    payload = {
        "local_axis": local_axis,
        "period_pixels": period_pixels,
        "sample_count": sample_count,
        "weight_count": weight_count,
        "index_count": index_count,
        "coverage": {
            "total": coverage["total"],
            "valid": coverage["valid"],
            "per_bin": list(coverage["per_bin"]),
        },
        "payload_sha256": payload_sha256,
    }
    return hashlib.sha256(SPLIT_PHASE_CANONICAL_JSON(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class SplitSeamProfileEvidence:
    """Frozen held-out D-only profile evidence (sixteen ordered fields)."""

    local_axis: str
    period_pixels: float
    profile_samples_float64_le_base64: str
    profile_weights_float64_le_base64: str
    axis_bin_indices_int64_le_base64: str
    profile_samples_encoding: str
    profile_weights_encoding: str
    axis_bin_indices_encoding: str
    sample_count: int
    weight_count: int
    index_count: int
    coverage: dict
    payload_sha256: str
    d_profile_sha256: str
    D_a_sha256: str
    seam_evidence_sha256: str

    def __post_init__(self) -> None:
        if self.local_axis not in ("u", "v"):
            raise ValueError("seam_profile_payload_invalid")
        period = _split_finite_float(self.period_pixels)
        if period <= 0.0:
            raise ValueError("seam_profile_payload_invalid")
        if (self.profile_samples_encoding, self.profile_weights_encoding,
                self.axis_bin_indices_encoding) != _SPLIT_PHASE_SAMPLE_ENCODINGS:
            raise ValueError("seam_profile_payload_invalid")
        for name in ("sample_count", "weight_count", "index_count"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError("seam_profile_payload_invalid")
        if self.sample_count < 1:
            raise ValueError("seam_profile_payload_invalid")
        if not (self.weight_count == self.index_count == self.sample_count):
            raise ValueError("seam_profile_payload_invalid")
        samples_raw, samples = _split_decode_array(
            self.profile_samples_float64_le_base64, "<f8")
        weights_raw, weights = _split_decode_array(
            self.profile_weights_float64_le_base64, "<f8")
        indices_raw, indices = _split_decode_array(
            self.axis_bin_indices_int64_le_base64, "<i8")
        if (len(samples) != self.sample_count or len(weights) != self.weight_count
                or len(indices) != self.index_count):
            raise ValueError("seam_profile_payload_invalid")
        if not bool(np.all(np.isfinite(samples))):
            raise ValueError("seam_profile_payload_invalid")
        if not bool(np.all(np.isfinite(weights))) or bool(np.any(weights < 0.0)):
            raise ValueError("negative_weight")
        integer_weights = []
        for value in weights:
            if float(value) != float(int(value)):
                raise ValueError("seam_profile_payload_invalid")
            integer_weights.append(int(value))
        if not np.array_equal(indices, np.arange(len(indices), dtype="<i8")):
            raise ValueError("seam_profile_payload_invalid")
        if (not isinstance(self.coverage, dict)
                or set(self.coverage) != {"total", "valid", "per_bin"}):
            raise ValueError("seam_profile_payload_invalid")
        total, valid, per_bin = (self.coverage["total"], self.coverage["valid"],
                                 self.coverage["per_bin"])
        if (type(total) is not int or type(valid) is not int
                or total < 0 or valid < 0 or valid > total
                or not isinstance(per_bin, tuple) or len(per_bin) != self.sample_count
                or any(type(value) is not int or value < 0 or value > valid
                       for value in per_bin)
                or sum(per_bin) != valid
                or tuple(integer_weights) != per_bin):
            raise ValueError("seam_profile_payload_invalid")
        if _split_payload_sha256(samples_raw, weights_raw, indices_raw) != self.payload_sha256:
            raise ValueError("seam_profile_payload_invalid")
        expected_d = _split_d_profile_sha256(
            self.local_axis, period, self.sample_count, self.weight_count,
            self.index_count, self.coverage, self.payload_sha256)
        if expected_d != self.d_profile_sha256 or not _split_hex64(self.D_a_sha256):
            raise ValueError("seam_profile_payload_invalid")
        object.__setattr__(self, "period_pixels", period)
        object.__setattr__(self, "coverage", {
            "total": total, "valid": valid, "per_bin": tuple(per_bin)})
        if (not _split_hex64(self.seam_evidence_sha256)
                or _split_digest(self.to_record(), "seam_evidence_sha256")
                != self.seam_evidence_sha256):
            raise ValueError("split_not_reconstructible")

    def to_record(self) -> dict:
        return {
            "local_axis": self.local_axis,
            "period_pixels": self.period_pixels,
            "profile_samples_float64_le_base64": self.profile_samples_float64_le_base64,
            "profile_weights_float64_le_base64": self.profile_weights_float64_le_base64,
            "axis_bin_indices_int64_le_base64": self.axis_bin_indices_int64_le_base64,
            "profile_samples_encoding": self.profile_samples_encoding,
            "profile_weights_encoding": self.profile_weights_encoding,
            "axis_bin_indices_encoding": self.axis_bin_indices_encoding,
            "sample_count": self.sample_count,
            "weight_count": self.weight_count,
            "index_count": self.index_count,
            "coverage": {
                "total": self.coverage["total"],
                "valid": self.coverage["valid"],
                "per_bin": list(self.coverage["per_bin"]),
            },
            "payload_sha256": self.payload_sha256,
            "d_profile_sha256": self.d_profile_sha256,
            "D_a_sha256": self.D_a_sha256,
            "seam_evidence_sha256": self.seam_evidence_sha256,
        }


def _make_seam_profile_evidence(local_axis, period_pixels, samples, weights,
                                indices, coverage, D_a_sha256) -> SplitSeamProfileEvidence:
    samples_raw = np.asarray(samples, dtype="<f8").tobytes(order="C")
    weights_raw = np.asarray(weights, dtype="<f8").tobytes(order="C")
    indices_raw = np.asarray(indices, dtype="<i8").tobytes(order="C")
    payload_sha = _split_payload_sha256(samples_raw, weights_raw, indices_raw)
    sample_count = int(len(samples_raw) // 8)
    d_profile_sha = _split_d_profile_sha256(
        local_axis, float(period_pixels), sample_count, sample_count,
        sample_count, coverage, payload_sha)
    payload = {
        "local_axis": local_axis,
        "period_pixels": float(period_pixels),
        "profile_samples_float64_le_base64": base64.b64encode(samples_raw).decode("ascii"),
        "profile_weights_float64_le_base64": base64.b64encode(weights_raw).decode("ascii"),
        "axis_bin_indices_int64_le_base64": base64.b64encode(indices_raw).decode("ascii"),
        "profile_samples_encoding": "float64-le-base64",
        "profile_weights_encoding": "float64-le-base64",
        "axis_bin_indices_encoding": "int64-le-base64",
        "sample_count": sample_count,
        "weight_count": sample_count,
        "index_count": sample_count,
        "coverage": {
            "total": coverage["total"], "valid": coverage["valid"],
            "per_bin": tuple(coverage["per_bin"])},
        "payload_sha256": payload_sha,
        "d_profile_sha256": d_profile_sha,
        "D_a_sha256": D_a_sha256,
    }
    return SplitSeamProfileEvidence(
        **payload,
        seam_evidence_sha256=_split_digest(payload, "seam_evidence_sha256"))


def _split_reading_identity(value):
    if (not isinstance(value, dict)
            or set(value) != {"frame", "plane_index", "reading_axis"}):
        return None
    if (type(value["frame"]) is not str or not value["frame"]
            or type(value["plane_index"]) is not int or value["plane_index"] < 0
            or type(value["reading_axis"]) is not int
            or value["reading_axis"] not in (0, 1, 2)):
        return None
    return {"frame": value["frame"], "plane_index": value["plane_index"],
            "reading_axis": value["reading_axis"]}


def _split_producer_input_sha256(producer_version_hex, geometry_sha256,
                                 reading_identity, patch_sha256,
                                 partition_config_sha256, budget_sha256) -> str:
    payload = {
        "producer_version_hex": producer_version_hex,
        "geometry_sha256": geometry_sha256,
        "reading_identity": reading_identity,
        "patch_sha256": patch_sha256,
        "partition_config_sha256": partition_config_sha256,
        "budget_sha256": budget_sha256,
    }
    return hashlib.sha256(SPLIT_PHASE_CANONICAL_JSON(payload)).hexdigest()


_SPLIT_EVIDENCE_FIELDS = ("lane_sets", "generation_evidence", "seam_profile_evidence")
_SPLIT_COORDINATE_FIELDS = ("G_u_sha256", "G_v_sha256", "D_u_sha256",
                            "D_v_sha256", "X_u_sha256", "X_v_sha256")


@dataclass(frozen=True, slots=True)
class SplitPhaseProducerResult:
    """Frozen atomic producer result (twenty-four ordered fields)."""

    status: str
    producer_version_hex: str
    producer_version_sha256: str
    geometry: SplitPhaseGeometry | None
    reading_identity: dict | None
    patch_ref: SplitPhasePatchRef | None
    partition_config_sha256: str | None
    lane_sets: tuple | None
    generation_evidence: tuple | None
    seam_profile_evidence: tuple | None
    planned_work: SplitPhaseProducerWork | None
    runtime_work: SplitPhaseProducerWork | None
    budget_stage: str | None
    refusals: tuple
    selector_salt_sha256: str | None
    partition_sha256: str | None
    G_u_sha256: str | None
    G_v_sha256: str | None
    D_u_sha256: str | None
    D_v_sha256: str | None
    X_u_sha256: str | None
    X_v_sha256: str | None
    producer_input_sha256: str | None
    result_sha256: str

    def __post_init__(self) -> None:
        if self.status not in ("complete", "refused", "budget_exhausted"):
            raise ValueError("split_not_reconstructible")
        if (self.producer_version_hex != SPLIT_PHASE_PRODUCER_VERSION.hex()
                or self.producer_version_sha256
                != hashlib.sha256(SPLIT_PHASE_PRODUCER_VERSION).hexdigest()):
            raise ValueError("split_not_reconstructible")
        if self.budget_stage not in (None, "planned"):
            raise ValueError("budget_stage_invalid")
        if self.reading_identity is not None:
            identity = _split_reading_identity(self.reading_identity)
            if identity is None:
                raise ValueError("split_not_reconstructible")
            object.__setattr__(self, "reading_identity", identity)
        if self.geometry is not None and not isinstance(self.geometry, SplitPhaseGeometry):
            raise ValueError("split_not_reconstructible")
        if self.patch_ref is not None and not isinstance(self.patch_ref, SplitPhasePatchRef):
            raise ValueError("split_not_reconstructible")
        for name, expected in (("lane_sets", SplitPhaseLaneSet),
                               ("generation_evidence", SplitGenerationPhaseEvidence),
                               ("seam_profile_evidence", SplitSeamProfileEvidence)):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, tuple)
                    or any(not isinstance(item, expected) for item in value)):
                raise ValueError("split_not_reconstructible")
        for name in ("planned_work", "runtime_work"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, SplitPhaseProducerWork):
                raise ValueError("split_not_reconstructible")
        if not isinstance(self.refusals, tuple) or any(
                not isinstance(value, SplitPhaseRefusal) for value in self.refusals):
            raise ValueError("split_not_reconstructible")
        for name in _SPLIT_COORDINATE_FIELDS:
            value = getattr(self, name)
            if value is not None and not _split_hex64(value):
                raise ValueError("split_not_reconstructible")
        for name in ("partition_config_sha256", "selector_salt_sha256",
                     "partition_sha256", "producer_input_sha256"):
            value = getattr(self, name)
            if value is not None and not _split_hex64(value):
                raise ValueError("split_not_reconstructible")
        self._validate_branch()
        if (not _split_hex64(self.result_sha256)
                or _split_digest(self.to_record(), "result_sha256")
                != self.result_sha256):
            raise ValueError("split_not_reconstructible")

    def _validate_branch(self) -> None:
        forbidden = (("geometry", "patch_ref") + _SPLIT_EVIDENCE_FIELDS
                     + _SPLIT_COORDINATE_FIELDS
                     + ("selector_salt_sha256", "partition_sha256"))
        if self.status == "complete":
            if self.budget_stage is not None or self.refusals:
                raise ValueError("split_not_reconstructible")
            if (self.geometry is None or self.reading_identity is None
                    or self.patch_ref is None or self.partition_config_sha256 is None
                    or self.planned_work is None or self.runtime_work is None
                    or self.producer_input_sha256 is None
                    or any(getattr(self, name) is None
                           for name in ("selector_salt_sha256", "partition_sha256")
                           + _SPLIT_COORDINATE_FIELDS)):
                raise ValueError("split_not_reconstructible")
            for name in _SPLIT_EVIDENCE_FIELDS:
                value = getattr(self, name)
                if value is None or len(value) == 0:
                    raise ValueError("split_not_reconstructible")
            if len(self.lane_sets) not in (4, 6) or len(self.generation_evidence) != 1 \
                    or len(self.seam_profile_evidence) != 2:
                raise ValueError("split_not_reconstructible")
            expected_roles = (
                (("u", "G"), ("u", "D"), ("v", "G"), ("v", "D"))
                if len(self.lane_sets) == 4 else
                (("u", "G"), ("u", "guard"), ("u", "D"),
                 ("v", "G"), ("v", "guard"), ("v", "D")))
            if tuple((item.local_axis, item.role)
                     for item in self.lane_sets) != expected_roles:
                raise ValueError("split_not_reconstructible")
            for item in self.lane_sets:
                role_key = {"G": "G", "D": "D", "guard": "X"}[item.role]
                if getattr(self, role_key + "_" + item.local_axis + "_sha256") \
                        != item.coordinate_sha256:
                    raise ValueError("split_not_reconstructible")
            generation = self.generation_evidence[0]
            if (generation.partition_sha256 != self.partition_sha256
                    or generation.G_u_sha256 != self.G_u_sha256
                    or generation.G_v_sha256 != self.G_v_sha256):
                raise ValueError("split_not_reconstructible")
            seam_u, seam_v = self.seam_profile_evidence
            if (seam_u.local_axis != "u" or seam_v.local_axis != "v"
                    or seam_u.D_a_sha256 != self.D_u_sha256
                    or seam_v.D_a_sha256 != self.D_v_sha256):
                raise ValueError("split_not_reconstructible")
            if self.runtime_work != self.planned_work:
                raise ValueError("work_over_cap")
        elif self.status == "refused":
            if not self.refusals or self.budget_stage is not None:
                raise ValueError("split_not_reconstructible")
            if any(getattr(self, name) is not None for name in forbidden):
                raise ValueError("split_not_reconstructible")
            if self.planned_work != self.runtime_work:
                raise ValueError("work_over_cap")
        else:
            if (len(self.refusals) != 1
                    or self.refusals[0].reason != "budget_exhausted"
                    or self.partition_config_sha256 is None
                    or self.planned_work is None or self.producer_input_sha256 is None
                    or self.reading_identity is not None
                    or self.budget_stage != "planned"
                    or self.runtime_work is not None):
                raise ValueError("split_not_reconstructible")
            if any(getattr(self, name) is not None for name in forbidden):
                raise ValueError("split_not_reconstructible")
            self._validate_budget_exhausted_detail()

    def _validate_budget_exhausted_detail(self) -> None:
        """Frozen ``budget_exhausted`` refusal-detail/work invariant (§6, §105).

        Only the single planned over-cap refusal is representable: the detail has
        exactly the four frozen keys, names a real cap that maps one-to-one to a
        ``planned_work`` counter, carries a legal positive exact-int (non-``bool``)
        limit, and its observed value is exactly that counter and strictly
        exceeds the limit.  An arbitrary, missing, extra, mistyped, under-cap, or
        non-exceeding detail is ``split_not_reconstructible``.
        """
        detail = self.refusals[0].detail
        if type(detail) is not dict or set(detail) != {
                "budget_stage", "cap", "limit", "observed"}:
            raise ValueError("split_not_reconstructible")
        if detail["budget_stage"] != "planned":
            raise ValueError("split_not_reconstructible")
        cap = detail["cap"]
        if type(cap) is not str or cap not in _SPLIT_CAP_TO_WORK:
            raise ValueError("split_not_reconstructible")
        limit = detail["limit"]
        if type(limit) is not int or limit < 1:
            raise ValueError("split_not_reconstructible")
        observed = getattr(self.planned_work, _SPLIT_CAP_TO_WORK[cap])
        if type(detail["observed"]) is not int or detail["observed"] != observed:
            raise ValueError("split_not_reconstructible")
        if observed <= limit:
            raise ValueError("split_not_reconstructible")

    def to_record(self) -> dict:
        return {
            "status": self.status,
            "producer_version_hex": self.producer_version_hex,
            "producer_version_sha256": self.producer_version_sha256,
            "geometry": None if self.geometry is None else self.geometry.to_record(),
            "reading_identity": (None if self.reading_identity is None
                                 else dict(self.reading_identity)),
            "patch_ref": None if self.patch_ref is None else self.patch_ref.to_record(),
            "partition_config_sha256": self.partition_config_sha256,
            "lane_sets": (None if self.lane_sets is None
                          else [value.to_record() for value in self.lane_sets]),
            "generation_evidence": (None if self.generation_evidence is None else
                                    [value.to_record()
                                     for value in self.generation_evidence]),
            "seam_profile_evidence": (None if self.seam_profile_evidence is None else
                                      [value.to_record()
                                       for value in self.seam_profile_evidence]),
            "planned_work": (None if self.planned_work is None
                             else self.planned_work.to_record()),
            "runtime_work": (None if self.runtime_work is None
                             else self.runtime_work.to_record()),
            "budget_stage": self.budget_stage,
            "refusals": [value.to_record() for value in self.refusals],
            "selector_salt_sha256": self.selector_salt_sha256,
            "partition_sha256": self.partition_sha256,
            "G_u_sha256": self.G_u_sha256,
            "G_v_sha256": self.G_v_sha256,
            "D_u_sha256": self.D_u_sha256,
            "D_v_sha256": self.D_v_sha256,
            "X_u_sha256": self.X_u_sha256,
            "X_v_sha256": self.X_v_sha256,
            "producer_input_sha256": self.producer_input_sha256,
            "result_sha256": self.result_sha256,
        }

    def to_json_bytes(self) -> bytes:
        return SPLIT_PHASE_CANONICAL_JSON(self.to_record())


def _make_result(payload: dict) -> SplitPhaseProducerResult:
    fields = {name: payload.get(name) for name in (
        "status", "producer_version_hex", "producer_version_sha256", "geometry",
        "reading_identity", "patch_ref", "partition_config_sha256", "lane_sets",
        "generation_evidence", "seam_profile_evidence", "planned_work",
        "runtime_work", "budget_stage", "refusals", "selector_salt_sha256",
        "partition_sha256", "G_u_sha256", "G_v_sha256", "D_u_sha256",
        "D_v_sha256", "X_u_sha256", "X_v_sha256", "producer_input_sha256")}
    fields["refusals"] = tuple(fields["refusals"] or ())
    digest_payload = dict(fields)
    digest_payload["geometry"] = (None if fields["geometry"] is None
                                  else fields["geometry"].to_record())
    digest_payload["reading_identity"] = fields["reading_identity"]
    digest_payload["patch_ref"] = (None if fields["patch_ref"] is None
                                   else fields["patch_ref"].to_record())
    digest_payload["lane_sets"] = (None if fields["lane_sets"] is None else
                                   [value.to_record() for value in fields["lane_sets"]])
    digest_payload["generation_evidence"] = (
        None if fields["generation_evidence"] is None else
        [value.to_record() for value in fields["generation_evidence"]])
    digest_payload["seam_profile_evidence"] = (
        None if fields["seam_profile_evidence"] is None else
        [value.to_record() for value in fields["seam_profile_evidence"]])
    digest_payload["planned_work"] = (None if fields["planned_work"] is None
                                      else fields["planned_work"].to_record())
    digest_payload["runtime_work"] = (None if fields["runtime_work"] is None
                                      else fields["runtime_work"].to_record())
    digest_payload["refusals"] = [value.to_record() for value in fields["refusals"]]
    fields["result_sha256"] = _split_digest(digest_payload, "result_sha256")
    return SplitPhaseProducerResult(**fields)


# --- Declarative calibration / grid data contracts (no fitting here) --------

SplitPhaseCalibrationTuple = tuple[
    tuple[int, int], tuple[int, int], tuple[int, int], int, int]
"""The five-element ordered JSON array of §7."""


def _split_dyadic_lt(first, second) -> bool:
    return first[0] * second[1] < second[0] * first[1]


def _split_calibration_tuple_lt(first, second) -> bool:
    for index in range(3):
        if _split_dyadic_lt(first[index], second[index]):
            return True
        if _split_dyadic_lt(second[index], first[index]):
            return False
    return (first[3], first[4]) < (second[3], second[4])


def _split_validate_calibration_tuple(value, *, tau_band_zero=True):
    if not isinstance(value, tuple) or len(value) != 5:
        raise ValueError("no_feasible_calibration")
    tau_band = _split_reduced_dyadic(value[0], allow_zero=tau_band_zero,
                                     allow_one=True)
    tau_margin = _split_reduced_dyadic(value[1], allow_zero=False, allow_one=True,
                                       require_positive=True)
    tau_energy = _split_reduced_dyadic(value[2], allow_zero=True, allow_one=True)
    if type(value[3]) is not int or value[3] < 1:
        raise ValueError("no_feasible_calibration")
    if type(value[4]) is not int or value[4] < 2:
        raise ValueError("no_feasible_calibration")
    return (tau_band, tau_margin, tau_energy, value[3], value[4])


@dataclass(frozen=True, slots=True)
class SplitPhaseCalibrationGrid:
    """Evaluation-only preregistered grid (seven ordered fields)."""

    grid_version: str
    evaluation_only: bool
    target_power: tuple[int, int]
    tuples: tuple
    null_control_ids: tuple
    seam_control_ids: tuple
    grid_sha256: str

    def __post_init__(self) -> None:
        if self.grid_version != "v2b-p0-calibration-grid-1" or self.evaluation_only is not True:
            raise ValueError("split_not_reconstructible")
        object.__setattr__(self, "target_power", _split_reduced_dyadic(
            self.target_power, allow_zero=False, allow_one=True,
            require_positive=True))
        if (not isinstance(self.tuples, tuple) or not self.tuples
                or len(self.tuples) > 4096):
            raise ValueError("split_not_reconstructible")
        validated = tuple(_split_validate_calibration_tuple(value)
                          for value in self.tuples)
        if any(not _split_calibration_tuple_lt(first, second)
               for first, second in zip(validated, validated[1:])):
            raise ValueError("split_not_reconstructible")
        for name in ("null_control_ids", "seam_control_ids"):
            value = getattr(self, name)
            if (not isinstance(value, tuple) or not value
                    or any(not _split_hex64(item) for item in value)):
                raise ValueError("split_not_reconstructible")
        object.__setattr__(self, "tuples", validated)
        if (not _split_hex64(self.grid_sha256)
                or _split_digest(self.to_record(), "grid_sha256") != self.grid_sha256):
            raise ValueError("split_not_reconstructible")

    def to_record(self) -> dict:
        return {
            "grid_version": self.grid_version,
            "evaluation_only": self.evaluation_only,
            "target_power": list(self.target_power),
            "tuples": [
                [list(value[0]), list(value[1]), list(value[2]), value[3], value[4]]
                for value in self.tuples],
            "null_control_ids": list(self.null_control_ids),
            "seam_control_ids": list(self.seam_control_ids),
            "grid_sha256": self.grid_sha256,
        }


@dataclass(frozen=True, slots=True)
class SplitPhaseCalibration:
    """Frozen declarative calibration record (no fitting performed here)."""

    calibration_version: str
    partition_config_sha256: str
    tau_band: tuple[int, int]
    tau_margin: tuple[int, int]
    tau_energy: tuple[int, int]
    min_held_out_samples: int
    min_distinct_frames_per_axis: int
    tie_break: str
    target_power: tuple[int, int]
    calibration_corpus_sha256: str
    evaluation_corpus_sha256: str
    calibration_sample_sha256s: tuple
    evaluation_sample_sha256s: tuple
    null_control_ids: tuple
    seam_control_ids: tuple
    calibration_grid_sha256: str
    calibration_sha256: str

    def __post_init__(self) -> None:
        if (self.calibration_version != "v2b-p0-calibration-1"
                or self.tie_break != "no_tie_break_abstain"
                or self.partition_config_sha256 != SPLIT_PHASE_PARTITION_CONFIG_SHA256):
            raise ValueError("split_not_reconstructible")
        object.__setattr__(self, "tau_band", _split_reduced_dyadic(
            self.tau_band, allow_zero=True, allow_one=True))
        object.__setattr__(self, "tau_margin", _split_reduced_dyadic(
            self.tau_margin, allow_zero=False, allow_one=True, require_positive=True))
        object.__setattr__(self, "tau_energy", _split_reduced_dyadic(
            self.tau_energy, allow_zero=True, allow_one=True))
        object.__setattr__(self, "target_power", _split_reduced_dyadic(
            self.target_power, allow_zero=False, allow_one=True, require_positive=True))
        if (type(self.min_held_out_samples) is not int or self.min_held_out_samples < 1
                or type(self.min_distinct_frames_per_axis) is not int
                or self.min_distinct_frames_per_axis < 2):
            raise ValueError("split_not_reconstructible")
        for name in ("calibration_corpus_sha256", "evaluation_corpus_sha256",
                     "calibration_grid_sha256"):
            if not _split_hex64(getattr(self, name)):
                raise ValueError("split_not_reconstructible")
        for name in ("calibration_sample_sha256s", "evaluation_sample_sha256s"):
            value = getattr(self, name)
            if not isinstance(value, tuple) or any(
                    not _split_hex64(item) for item in value):
                raise ValueError("split_not_reconstructible")
        for name in ("null_control_ids", "seam_control_ids"):
            value = getattr(self, name)
            if (not isinstance(value, tuple) or not value
                    or any(not _split_hex64(item) for item in value)):
                raise ValueError("split_not_reconstructible")
        if set(self.calibration_sample_sha256s) & set(self.evaluation_sample_sha256s):
            raise ValueError("split_not_reconstructible")
        if (not _split_hex64(self.calibration_sha256)
                or _split_digest(self.to_record(), "calibration_sha256")
                != self.calibration_sha256):
            raise ValueError("split_not_reconstructible")

    def to_record(self) -> dict:
        return {
            "calibration_version": self.calibration_version,
            "partition_config_sha256": self.partition_config_sha256,
            "tau_band": list(self.tau_band),
            "tau_margin": list(self.tau_margin),
            "tau_energy": list(self.tau_energy),
            "min_held_out_samples": self.min_held_out_samples,
            "min_distinct_frames_per_axis": self.min_distinct_frames_per_axis,
            "tie_break": self.tie_break,
            "target_power": list(self.target_power),
            "calibration_corpus_sha256": self.calibration_corpus_sha256,
            "evaluation_corpus_sha256": self.evaluation_corpus_sha256,
            "calibration_sample_sha256s": list(self.calibration_sample_sha256s),
            "evaluation_sample_sha256s": list(self.evaluation_sample_sha256s),
            "null_control_ids": list(self.null_control_ids),
            "seam_control_ids": list(self.seam_control_ids),
            "calibration_grid_sha256": self.calibration_grid_sha256,
            "calibration_sha256": self.calibration_sha256,
        }


_SPLIT_CAP_TO_WORK = {cap: work for work, cap in _SPLIT_WORK_TO_CAP.items()}


class _SplitRefusal(Exception):
    """Internal carrier for an atomic refused result."""

    def __init__(self, reason, detail=None, cap=None):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail
        self.cap = cap


def _split_work_within_cap(work, budget) -> bool:
    return all(getattr(work, name) <= getattr(budget, _SPLIT_WORK_TO_CAP[name])
               for name in _SPLIT_WORK_FIELDS)


def _split_first_over_cap(work, budget):
    for name in _SPLIT_WORK_FIELDS:
        if getattr(work, name) > getattr(budget, _SPLIT_WORK_TO_CAP[name]):
            return _SPLIT_WORK_TO_CAP[name]
    return None


def _split_validated_patch(weights_patch) -> np.ndarray:
    if weights_patch is None:
        raise _SplitRefusal("p0_patch_unavailable")
    try:
        patch = np.asarray(weights_patch)
        if patch.ndim != 2:
            raise _SplitRefusal("split_not_reconstructible")
        patch = patch.astype("<f8", copy=False)
    except _SplitRefusal:
        raise
    except (TypeError, ValueError, OverflowError) as error:
        raise _SplitRefusal("split_not_reconstructible") from error
    if patch.shape[0] < 1 or patch.shape[1] < 1:
        raise _SplitRefusal("split_not_reconstructible")
    if not bool(np.all(np.isfinite(patch))):
        raise _SplitRefusal("nonfinite_patch")
    if bool(np.any(patch < 0.0)):
        raise _SplitRefusal("negative_weight")
    return np.ascontiguousarray(patch, dtype="<f8")


def _split_coordinate_sha256(local_axis, lane_start, lane_stop, height, width) -> str:
    """Coordinate-set-only role digest: never reads any patch value (§4).

    The full ordered 2D lane coordinate set is enumerated row-major for ``u``
    (every row of the lane block, all ``W`` columns) and column-major for ``v``
    (every column of the lane block, all ``H`` rows). A lane coordinate change
    changes the digest even when the reduced axis-bin indices do not, while a
    value mutation at a fixed partition cannot change it.
    """
    if local_axis == "u":
        coordinates = [[row, col]
                       for row in range(int(lane_start), int(lane_stop))
                       for col in range(int(width))]
    else:
        coordinates = [[row, col]
                       for col in range(int(lane_start), int(lane_stop))
                       for row in range(int(height))]
    payload = {"local_axis": local_axis, "coordinates": coordinates}
    return hashlib.sha256(SPLIT_PHASE_CANONICAL_JSON(payload)).hexdigest()


def _split_lane_sets(plan_u, plan_v, height, width):
    records = []
    digests = {}
    for plan in (plan_u, plan_v):
        orth = width if plan.local_axis == "u" else height
        for role in ("G", "guard", "D"):
            if role == "G":
                start, stop = plan.g_start, plan.g_stop
            elif role == "D":
                start, stop = plan.d_start, plan.d_stop
            elif plan.guard_start is None:
                # The zero-width guard role is omitted from the lane sets but
                # its (empty-set) coordinate digest is still retained.
                start, stop = plan.nG, plan.nG
            else:
                start, stop = plan.guard_start, plan.guard_stop
            digest_key = {"G": "G", "D": "D", "guard": "X"}[role] + "_" + plan.local_axis + "_sha256"
            coordinate_sha = _split_coordinate_sha256(
                plan.local_axis, start, stop, height, width)
            digests[digest_key] = coordinate_sha
            if role == "guard" and plan.guard_start is None:
                continue
            if plan.local_axis == "u":
                row_range, column_range = (start, stop), None
            else:
                row_range, column_range = None, (start, stop)
            records.append(_make_lane_set(
                plan.local_axis, role, start, stop, row_range, column_range,
                (stop - start) * orth, coordinate_sha))
    return tuple(records), digests


def _split_partition_sha256(geometry, config, plan_u, plan_v, height, width) -> str:
    def axis_record(plan):
        return {
            "orientation": plan.orientation,
            "g": [plan.g_start, plan.g_stop],
            "guard": (None if plan.guard_start is None
                      else [plan.guard_start, plan.guard_stop]),
            "d": [plan.d_start, plan.d_stop],
        }
    payload = {
        "partition_version": config.partition_version,
        "selector_version_hex": config.selector_version.hex(),
        "selector_salt": config.selector_salt,
        "shape": [int(height), int(width)],
        "geometry": geometry.to_record(),
        "axes": {"u": axis_record(plan_u), "v": axis_record(plan_v)},
    }
    return hashlib.sha256(SPLIT_PHASE_CANONICAL_JSON(payload)).hexdigest()


def _split_g_profile_sha256(g_u, g_v) -> str:
    payload = {"u": [float(value) for value in g_u],
               "v": [float(value) for value in g_v]}
    return hashlib.sha256(SPLIT_PHASE_CANONICAL_JSON(payload)).hexdigest()


def _split_planned_counters(height, width, plan_u, plan_v, inline_limit):
    guard_present = plan_u.guard_start is not None or plan_v.guard_start is not None
    lane_sets = 6 if guard_present else 4
    return {
        "lanes_per_axis": max(int(height), int(width)),
        "patch_samples": int(height) * int(width),
        "inline_patch_samples": (int(height) * int(width)
                                 if int(height) * int(width) <= int(inline_limit) else 0),
        "g_samples_per_axis": max((plan_u.g_stop - plan_u.g_start) * width,
                                  (plan_v.g_stop - plan_v.g_start) * height),
        "d_samples_per_axis": max((plan_u.d_stop - plan_u.d_start) * width,
                                  (plan_v.d_stop - plan_v.d_start) * height),
        "guard_samples_per_axis": max(plan_u.guard_lanes * width,
                                      plan_v.guard_lanes * height),
        "profile_terms": int(width) + int(height),
        "d_profile_terms": 3 * (int(width) + int(height)),
        # Both axis records' fully framed byte sequences: the frozen payload
        # version bytes plus three uint64_le prefixes plus the three canonical
        # little-endian arrays (8*n each), with n = W for u and n = H for v.
        "payload_bytes": (2 * len(SPLIT_SEAM_PROFILE_PAYLOAD_VERSION) + 48
                          + 24 * (int(width) + int(height))),
        # Each present lane set plus one generation record plus two seam records.
        "source_records": lane_sets + 3,
        # One result envelope plus every term of source_records.
        "replay_checks": lane_sets + 4,
    }


def _split_budget_result(cap, budget, planned, config_sha, input_sha):
    """Atomic planned-only exhaustion: exact over-cap plan, no runtime stage."""
    counter = _SPLIT_CAP_TO_WORK[cap]
    detail = {
        "budget_stage": "planned",
        "cap": cap,
        "limit": getattr(budget, cap),
        "observed": getattr(planned, counter),
    }
    return _make_result({
        "status": "budget_exhausted",
        "producer_version_hex": SPLIT_PHASE_PRODUCER_VERSION.hex(),
        "producer_version_sha256": hashlib.sha256(SPLIT_PHASE_PRODUCER_VERSION).hexdigest(),
        "geometry": None,
        "reading_identity": None,
        "patch_ref": None,
        "partition_config_sha256": config_sha,
        "lane_sets": None,
        "generation_evidence": None,
        "seam_profile_evidence": None,
        "planned_work": planned,
        "runtime_work": None,
        "budget_stage": "planned",
        "refusals": (_make_refusal("budget_exhausted", detail, cap=cap),),
        "selector_salt_sha256": None,
        "partition_sha256": None,
        "G_u_sha256": None,
        "G_v_sha256": None,
        "D_u_sha256": None,
        "D_v_sha256": None,
        "X_u_sha256": None,
        "X_v_sha256": None,
        "producer_input_sha256": input_sha,
    })


def _split_simple_refused(reason):
    return _make_result({
        "status": "refused",
        "producer_version_hex": SPLIT_PHASE_PRODUCER_VERSION.hex(),
        "producer_version_sha256": hashlib.sha256(SPLIT_PHASE_PRODUCER_VERSION).hexdigest(),
        "geometry": None,
        "reading_identity": None,
        "patch_ref": None,
        "partition_config_sha256": None,
        "lane_sets": None,
        "generation_evidence": None,
        "seam_profile_evidence": None,
        "planned_work": None,
        "runtime_work": None,
        "budget_stage": None,
        "refusals": (_make_refusal(reason),),
        "selector_salt_sha256": None,
        "partition_sha256": None,
        "G_u_sha256": None,
        "G_v_sha256": None,
        "D_u_sha256": None,
        "D_v_sha256": None,
        "X_u_sha256": None,
        "X_v_sha256": None,
        "producer_input_sha256": None,
    })


def _split_core(weights_patch, *, geometry, frame, plane_index, reading_axis,
                source_region_id, partition_config, budget):
    producer_version_hex = SPLIT_PHASE_PRODUCER_VERSION.hex()
    producer_version_sha = hashlib.sha256(SPLIT_PHASE_PRODUCER_VERSION).hexdigest()
    state = {"identity": None, "config_sha": None, "planned": None,
             "budget": budget, "input_sha": None}

    def refuse(reason, detail=None, cap=None):
        planned = state["planned"]
        runtime = None
        if planned is not None and _split_work_within_cap(planned, budget):
            runtime = planned
        else:
            planned = None
        return _make_result({
            "status": "refused",
            "producer_version_hex": producer_version_hex,
            "producer_version_sha256": producer_version_sha,
            "geometry": None,
            "reading_identity": state["identity"],
            "patch_ref": None,
            "partition_config_sha256": state["config_sha"],
            "lane_sets": None,
            "generation_evidence": None,
            "seam_profile_evidence": None,
            "planned_work": planned,
            "runtime_work": runtime,
            "budget_stage": None,
            "refusals": (_make_refusal(reason, detail, cap=cap),),
            "selector_salt_sha256": None,
            "partition_sha256": None,
            "G_u_sha256": None,
            "G_v_sha256": None,
            "D_u_sha256": None,
            "D_v_sha256": None,
            "X_u_sha256": None,
            "X_v_sha256": None,
            "producer_input_sha256": state["input_sha"],
        })

    try:
        if not isinstance(geometry, SplitPhaseGeometry):
            raise _SplitRefusal("split_not_reconstructible")
        if not isinstance(budget, SplitPhaseProducerBudget):
            raise _SplitRefusal("budget_stage_invalid")
        if not isinstance(partition_config, SplitPhasePartitionConfig):
            raise _SplitRefusal("selector_domain_invalid")
        config_sha = partition_config_sha256(partition_config)
        state["config_sha"] = config_sha
        if config_sha != SPLIT_PHASE_PARTITION_CONFIG_SHA256:
            raise _SplitRefusal("partition_config_not_builtin")
        identity = _split_reading_identity({
            "frame": frame, "plane_index": plane_index,
            "reading_axis": reading_axis})
        if identity is None:
            raise _SplitRefusal("split_not_reconstructible")
        state["identity"] = identity
        if source_region_id is not None and not _split_hex64(source_region_id):
            raise _SplitRefusal("split_not_reconstructible")
        patch = _split_validated_patch(weights_patch)
        height, width = int(patch.shape[0]), int(patch.shape[1])
        raw = patch.tobytes(order="C")
        patch_sha = _split_patch_sha256(height, width, len(raw), raw)
        state["input_sha"] = _split_producer_input_sha256(
            producer_version_hex, _split_geometry_sha256(geometry), identity,
            patch_sha, config_sha, budget.budget_sha256)
        plan_u = split_phase_lane_plan("u", height, width, partition_config)
        if isinstance(plan_u, SplitPhaseRefusal):
            raise _SplitRefusal(plan_u.reason, plan_u.detail)
        plan_v = split_phase_lane_plan("v", width, height, partition_config)
        if isinstance(plan_v, SplitPhaseRefusal):
            raise _SplitRefusal(plan_v.reason, plan_v.detail)
        planned = _make_work(_split_planned_counters(
            height, width, plan_u, plan_v, budget.max_inline_patch_samples))
        state["planned"] = planned
        trigger = _split_first_over_cap(planned, budget)
        if trigger is not None:
            return _split_budget_result(trigger, budget, planned,
                                        config_sha, state["input_sha"])
        lane_sets, digests = _split_lane_sets(plan_u, plan_v, height, width)
        partition_sha = _split_partition_sha256(geometry, partition_config,
                                                plan_u, plan_v, height, width)
        selector_salt_sha = hashlib.sha256(
            SPLIT_PHASE_CANONICAL_JSON(partition_config.selector_salt)).hexdigest()
        g_u = patch[plan_u.g_start:plan_u.g_stop, :].sum(axis=0)
        g_v = patch[:, plan_v.g_start:plan_v.g_stop].sum(axis=1)
        g_profile_sha = _split_g_profile_sha256(g_u, g_v)
        measured_u = _periodic_energy_axis(g_u, geometry.period_pixels)
        measured_v = _periodic_energy_axis(g_v, geometry.period_pixels)
        generation = (
            _make_generation_evidence({
                "frame": identity["frame"],
                "plane_index": identity["plane_index"],
                "reading_axis": identity["reading_axis"],
                "period_pixels": geometry.period_pixels,
                "patch_centre_uv": geometry.centre,
                "extent": geometry.extent,
                "samples": geometry.samples,
                "source_region_id": source_region_id,
                "phase_uv_pixels": (measured_u[0], measured_v[0]),
                "response_magnitude_uv": (measured_u[1], measured_v[1]),
                "profile_mass_uv": (measured_u[2], measured_v[2]),
                "concentration_uv": (measured_u[3], measured_v[3]),
                "unavailable_reason_uv": (measured_u[4], measured_v[4]),
                "g_profile_sha256": g_profile_sha,
                "partition_sha256": partition_sha,
                "G_u_sha256": digests["G_u_sha256"],
                "G_v_sha256": digests["G_v_sha256"],
                "G_u_sample_count": (plan_u.g_stop - plan_u.g_start) * width,
                "G_v_sample_count": (plan_v.g_stop - plan_v.g_start) * height,
            }),
        )
        seam = (
            _make_seam_profile_evidence(
                "u", geometry.period_pixels,
                patch[plan_u.d_start:plan_u.d_stop, :].sum(axis=0),
                np.full(width, plan_u.d_stop - plan_u.d_start, dtype="<f8"),
                np.arange(width, dtype="<i8"),
                {"total": (plan_u.d_stop - plan_u.d_start) * width,
                 "valid": (plan_u.d_stop - plan_u.d_start) * width,
                 "per_bin": tuple([plan_u.d_stop - plan_u.d_start] * width)},
                digests["D_u_sha256"]),
            _make_seam_profile_evidence(
                "v", geometry.period_pixels,
                patch[:, plan_v.d_start:plan_v.d_stop].sum(axis=1),
                np.full(height, plan_v.d_stop - plan_v.d_start, dtype="<f8"),
                np.arange(height, dtype="<i8"),
                {"total": (plan_v.d_stop - plan_v.d_start) * height,
                 "valid": (plan_v.d_stop - plan_v.d_start) * height,
                 "per_bin": tuple([plan_v.d_stop - plan_v.d_start] * height)},
                digests["D_v_sha256"]),
        )
        return _make_result({
            "status": "complete",
            "producer_version_hex": producer_version_hex,
            "producer_version_sha256": producer_version_sha,
            "geometry": geometry,
            "reading_identity": identity,
            "patch_ref": _make_patch_ref(patch, budget.max_inline_patch_samples),
            "partition_config_sha256": config_sha,
            "lane_sets": lane_sets,
            "generation_evidence": generation,
            "seam_profile_evidence": seam,
            "planned_work": planned,
            "runtime_work": planned,
            "budget_stage": None,
            "refusals": (),
            "selector_salt_sha256": selector_salt_sha,
            "partition_sha256": partition_sha,
            "G_u_sha256": digests["G_u_sha256"],
            "G_v_sha256": digests["G_v_sha256"],
            "D_u_sha256": digests["D_u_sha256"],
            "D_v_sha256": digests["D_v_sha256"],
            "X_u_sha256": digests.get("X_u_sha256"),
            "X_v_sha256": digests.get("X_v_sha256"),
            "producer_input_sha256": state["input_sha"],
        })
    except _SplitRefusal as error:
        return refuse(error.reason, error.detail, cap=error.cap)
    except (ValueError, TypeError, KeyError, IndexError, AttributeError,
            OverflowError, FloatingPointError):
        return refuse("split_not_reconstructible")


def produce_split_phase_evidence(
        weights_patch, *, geometry: SplitPhaseGeometry, frame, plane_index,
        reading_axis, source_region_id,
        partition_config: SplitPhasePartitionConfig,
        budget: SplitPhaseProducerBudget,
        producer_version=SPLIT_PHASE_PRODUCER_VERSION) -> SplitPhaseProducerResult:
    """Frozen P0 producer: typed atomic result, never bytes and never IO."""
    if type(producer_version) is not bytes or producer_version != SPLIT_PHASE_PRODUCER_VERSION:
        return _split_simple_refused("split_not_reconstructible")
    result = _split_core(
        weights_patch, geometry=geometry, frame=frame, plane_index=plane_index,
        reading_axis=reading_axis, source_region_id=source_region_id,
        partition_config=partition_config, budget=budget)
    if result.status == "complete" and not validate_split_phase_evidence(
            result, weights_patch, geometry=geometry,
            partition_config=partition_config, budget=budget):
        return _split_simple_refused("split_not_reconstructible")
    return result


def validate_split_phase_evidence(
        record, weights_patch, *, geometry: SplitPhaseGeometry,
        partition_config: SplitPhasePartitionConfig,
        budget: SplitPhaseProducerBudget) -> bool:
    """Frozen bool replay: True only for a complete, byte-identical replay."""
    try:
        if not isinstance(record, SplitPhaseProducerResult) or record.status != "complete":
            return False
        if (not isinstance(geometry, SplitPhaseGeometry)
                or not isinstance(partition_config, SplitPhasePartitionConfig)
                or not isinstance(budget, SplitPhaseProducerBudget)):
            return False
        identity = record.reading_identity
        if identity is None or record.geometry is None or record.patch_ref is None:
            return False
        if record.geometry.to_record() != geometry.to_record():
            return False
        if (record.partition_config_sha256 != SPLIT_PHASE_PARTITION_CONFIG_SHA256
                or partition_config_sha256(partition_config)
                != SPLIT_PHASE_PARTITION_CONFIG_SHA256):
            return False
        if not record.generation_evidence or not record.seam_profile_evidence:
            return False
        source_region_id = record.generation_evidence[0].source_region_id
        patch = np.asarray(weights_patch)
        if patch.ndim != 2 or patch.shape[0] < 1 or patch.shape[1] < 1:
            return False
        patch = np.ascontiguousarray(patch.astype("<f8", copy=False))
        if (not bool(np.all(np.isfinite(patch))) or bool(np.any(patch < 0.0))
                or int(patch.shape[0]) != record.patch_ref.rows
                or int(patch.shape[1]) != record.patch_ref.columns):
            return False
        raw = patch.tobytes(order="C")
        if _split_patch_sha256(int(patch.shape[0]), int(patch.shape[1]),
                               len(raw), raw) != record.patch_ref.patch_sha256:
            return False
        rebuilt = _split_core(
            patch, geometry=geometry, frame=identity["frame"],
            plane_index=identity["plane_index"],
            reading_axis=identity["reading_axis"],
            source_region_id=source_region_id,
            partition_config=partition_config, budget=budget)
        if rebuilt.status != "complete":
            return False
        return (SPLIT_PHASE_CANONICAL_JSON(rebuilt.to_record())
                == SPLIT_PHASE_CANONICAL_JSON(record.to_record()))
    except Exception:
        return False
