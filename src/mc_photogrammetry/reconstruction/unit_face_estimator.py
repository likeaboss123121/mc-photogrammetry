"""Private single-plane numerical experiment, conditional on frozen image optics.

Distance is perpendicular camera-to-plane distance in blocks. Coordinates are
the two ascending world-axis indices other than the normal axis, at distance
one, from ``lattice`` rays. Texture coordinates are ``d*q + phase`` projected
onto catalogue raw TL->TR and TL->BL edges, plus one half block. Phases are
world-axis block-centre phases modulo one. No world origin or placement follows.

CalibrationResult and the partition conditioning digest are supplied upstream
image-derived evidence. Their image origin, exclusion of held-out/GUI pixels,
and optical uncertainty are not authenticated here. Reported intervals are
conditional deterministic residual-sublevel enclosures, not confidence levels.
The source support is replayed by the existing immutable witness contract.
Only its original source pixels are measured, once per role; rectified sampling
density never becomes a second independent observation of the same pixel.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import heapq
import json
import math

import numpy as np

from ..blocks.unit_face_catalogue import UnitFaceCatalogue, UnitFace
from .calibration import CalibrationResult, axis_directions
from . import lattice
from .unit_face_lattice import (PartitionedSupport, UnitFaceContractError,
                               WitnessProof, SourceTapPartition, observe_role_pixels)


PIXEL_WORK_VERSION = 'sampled-rgb-pixels-v1:(max_work+7)*Ng+(U+1)*Nv'


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _hex(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


@dataclass(frozen=True, slots=True)
class PlaneContext:
    axis: int
    axis_matrix: tuple[tuple[float, float, float], ...]
    focal_pixels: float
    principal_point: tuple[float, float]
    source_shape: tuple[int, int]
    conditioning_sha256: str
    calibration_sha256: str

    def __post_init__(self):
        matrix = np.asarray(self.axis_matrix, dtype=float)
        if (type(self.axis) is not int or self.axis not in (0, 1, 2)
                or type(self.axis_matrix) is not tuple
                or any(type(r) is not tuple for r in self.axis_matrix)
                or matrix.shape != (3, 3) or not np.isfinite(matrix).all()
                or not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-10)
                or not math.isfinite(self.focal_pixels) or self.focal_pixels <= 0
                or type(self.principal_point) is not tuple or len(self.principal_point) != 2
                or not all(math.isfinite(v) for v in self.principal_point)
                or type(self.source_shape) is not tuple or len(self.source_shape) != 2
                or not all(type(v) is int and v > 0 for v in self.source_shape)
                or not _hex(self.conditioning_sha256) or not _hex(self.calibration_sha256)):
            raise ValueError('invalid_plane_context')

    @property
    def sha256(self):
        return _hash(asdict(self))


def context_from_calibration(calibration, axis, *, conditioning_sha256):
    """Freeze existing image calibration; accepts no distance/XYZ/phase metadata."""
    if type(calibration) is not CalibrationResult or calibration.inliers < 6:
        raise ValueError('image_calibration_unavailable')
    matrix = axis_directions(calibration.yaw_degrees, calibration.pitch_degrees,
                             calibration.roll_degrees)
    centre = calibration.principal_point or (calibration.width / 2, calibration.height / 2)
    return PlaneContext(axis, tuple(tuple(float(x) for x in r) for r in matrix),
                        float(calibration.focal_pixels), tuple(centre),
                        (calibration.height, calibration.width), conditioning_sha256,
                        _hash(asdict(calibration)))


@dataclass(frozen=True, slots=True)
class SearchPolicy:
    """Preregister before observations; ceilings are lowerable, never adaptive.

    Thresholds are synthetic experiment policy, not tuned to any fixture. Gain
    and bias are per RGB channel, in normalized native RGB (no gamma estimate).
    Bounds cover one whole circular phase per axis and a supplied distance range.
    """
    distance_bounds: tuple[float, float] = (1.0, 8.0)
    max_work: int = 100_000
    max_catalogue_faces: int = 49_152
    max_compatible_faces: int = 8_192
    max_unique_templates: int = 8_192
    max_native_rgba_bytes: int = 67_108_864
    max_pixel_work: int = 250_000_000
    max_pixels_per_role: int = 2048
    gain_bounds: tuple[float, float] = (0.95, 1.05)
    bias_bounds: tuple[float, float] = (-0.02, 0.02)
    generation_rmse_limit: float = 0.025
    validation_rmse_limit: float = 0.03
    residual_slack: float = 0.004
    leaf_relative_distance: float = 0.005
    leaf_phase: float = 0.01
    min_pixels_per_role: int = 48
    min_incidence: float = 0.15
    min_pixels_per_texel: float = 2.0

    def __post_init__(self):
        for bounds in (self.distance_bounds, self.gain_bounds, self.bias_bounds):
            if (type(bounds) is not tuple or len(bounds) != 2
                    or not all(math.isfinite(x) for x in bounds) or bounds[0] >= bounds[1]):
                raise ValueError('invalid_search_bounds')
        if (self.distance_bounds[0] <= 0 or self.distance_bounds[1] > 64
                or self.gain_bounds[0] < .5 or self.gain_bounds[1] > 2
                or self.bias_bounds[0] < -.25 or self.bias_bounds[1] > .25):
            raise ValueError('invalid_search_bounds')
        for value, ceiling in ((self.max_work, 100_000),
                               (self.max_catalogue_faces, 49_152),
                               (self.max_compatible_faces, 8_192),
                               (self.max_unique_templates, 8_192),
                               (self.max_native_rgba_bytes, 67_108_864),
                               (self.max_pixel_work, 250_000_000),
                               (self.max_pixels_per_role, 2048)):
            if type(value) is not int or not 0 < value <= ceiling:
                raise ValueError('invalid_search_cap')
        if (type(self.min_pixels_per_role) is not int or self.min_pixels_per_role < 48
                or not 0 < self.leaf_relative_distance <= 0.005
                or not 0 < self.leaf_phase <= 0.01
                or not 0 < self.residual_slack <= 0.004
                or not 0 < self.generation_rmse_limit <= 0.025
                or not 0 < self.validation_rmse_limit <= 0.03
                or not 0.15 <= self.min_incidence <= 1
                or not math.isfinite(self.min_pixels_per_texel)
                or self.min_pixels_per_texel < 2):
            raise ValueError('invalid_search_threshold')

    @property
    def sha256(self):
        return _hash({'policy': asdict(self), 'pixel_work_version': PIXEL_WORK_VERSION})


@dataclass(frozen=True, slots=True)
class Candidate:
    face_sha256: str
    geometry_sha256: str
    state_aliases: tuple[str, ...]
    distance: float
    phase: tuple[float, float]
    gain: tuple[float, float, float]
    bias: tuple[float, float, float]
    generation_rmse: float
    validation_rmse: float | None
    validation_max_tile_rmse: float | None
    retained_geometry: bool | None = None  # None when complete coverage is unavailable


@dataclass(frozen=True, slots=True)
class ParameterBox:
    geometry_sha256: str
    distance: tuple[float, float]
    phase_u: tuple[float, float]  # closed unwrapped endpoints, not start/width
    phase_v: tuple[float, float]
    residual_lower_bound: float


@dataclass(frozen=True, slots=True)
class FitResult:
    status: str
    reason: str
    candidates: tuple[Candidate, ...]
    representative: Candidate | None
    cover: tuple[ParameterBox, ...]
    distance_interval: tuple[float, float] | None
    phase_intervals: tuple[tuple[float, float], tuple[float, float]] | None
    residual_lower_bound: float | None
    rival_separation: float | None
    conditioning: float | None
    work_used: int
    work_budget: int
    search_complete: bool
    retained_boxes: int
    residual_admission_threshold: float | None
    context_sha256: str
    support_sha256: str
    catalogue_sha256: str
    source_generation_sha256: str
    source_validation_sha256: str
    policy_sha256: str
    compatible_faces: int
    unique_templates: int
    native_rgba_bytes: int
    estimated_pixel_work: int
    actual_pixel_work: int
    pixel_work_budget: int
    domain_search_complete: bool
    search_policy: SearchPolicy | None
    pixel_work_version: str = PIXEL_WORK_VERSION
    placement_supported: bool = False

    def __post_init__(self):
        if self.status not in ('supported', 'abstained') or self.placement_supported:
            raise ValueError('invalid_fit_status')
        if self.status == 'supported' and (self.reason or not self.search_complete
                or self.distance_interval is None or self.phase_intervals is None
                or not self.candidates or self.representative is None
                or not self.domain_search_complete or self.work_used > self.work_budget
                or self.actual_pixel_work > self.pixel_work_budget
                or self.estimated_pixel_work > self.pixel_work_budget):
            raise ValueError('uncertified_fit')
        if self.status == 'abstained' and not self.reason:
            raise ValueError('missing_abstention_reason')

    def to_record(self):
        return asdict(self)


def _rays(context, indices):
    h, w = context.source_shape
    points = np.column_stack((np.asarray(indices) % w, np.asarray(indices) // w))
    mapped = lattice._map_rays_to_unit_plane(points, context.axis,
        np.asarray(context.axis_matrix), context.focal_pixels, (w, h),
        context.principal_point, require_all=True)
    if isinstance(mapped, lattice.PlaneRayAbstention):
        raise ValueError('grazing_or_mixed_plane')
    incidence = float(np.min(np.abs(mapped.incidence)))
    return mapped.coordinates, mapped.side, incidence


def _face_geometry(face, axis, native_digest=None):
    """Validate native byte/corner identity without allocating floating texture."""
    if type(face) is not UnitFace:
        raise ValueError('unsupported_asset')
    h, w = face.texture_size
    corners = np.asarray(face.texture_corners, dtype=float)
    if (not 2 <= h == w <= 128 or type(face.rgba) is not bytes
            or len(face.rgba) != h * w * 4 or face.uv != (0, 0, 16, 16)
            or corners.shape != (4, 3) or not np.isfinite(corners).all()
            or (native_digest or hashlib.sha256(face.rgba).hexdigest()) != face.rgba_sha256):
        raise ValueError('unsupported_asset')
    rgba = np.frombuffer(face.rgba, np.uint8).reshape(h, w, 4)
    if native_digest is None and not np.all(rgba[..., 3] == 255):
        raise ValueError('unsupported_asset')
    others = [a for a in range(3) if a != axis]
    edges = np.stack((corners[1] - corners[0], corners[3] - corners[0]))
    transform = edges[:, others]
    normal = {'west': (-1, 0, 0), 'east': (1, 0, 0), 'down': (0, -1, 0),
              'up': (0, 1, 0), 'north': (0, 0, -1), 'south': (0, 0, 1)}.get(face.face_slot)
    if (not np.all(np.isin(corners, (-0.5, 0.5)))
            or not np.all(corners[:, axis] == corners[0, axis])
            or not np.array_equal(transform @ transform.T, np.eye(2))
            or not np.all(edges[:, axis] == 0)
            or not np.array_equal(corners[2], corners[1] + corners[3] - corners[0])
            or normal is None or not np.array_equal(-np.cross(edges[0], edges[1]), normal)
            or corners[0, axis] != .5*normal[axis]):
        raise ValueError('unsupported_asset')
    geometry = _hash({'rgba': face.rgba_sha256, 'shape': face.texture_size,
                      'transform': transform.tolist(), 'face_slot': face.face_slot})
    return transform, geometry


def _native_texture(face):
    h, w = face.texture_size
    return np.frombuffer(face.rgba, np.uint8).reshape(h, w, 4)[..., :3].astype(float)/255


def _template(face, axis):
    transform, geometry = _face_geometry(face, axis)
    return _native_texture(face), transform, geometry


def _derivative_bounds(texture):
    h, w = texture.shape[:2]
    return (w*np.max(np.abs(np.roll(texture, -1, axis=1)-texture), axis=(0, 1)),
            h*np.max(np.abs(np.roll(texture, -1, axis=0)-texture), axis=(0, 1)))


class _BudgetAbort(Exception):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


@dataclass(slots=True)
class _WorkBudget:
    policy: SearchPolicy
    evaluations: int = 0
    pixels: int = 0

    def charge(self, pixels, *, evaluation=False, stage='search'):
        """Reserve before sampling; search leaves one model fit for midpoint."""
        if evaluation and self.evaluations >= self.policy.max_work - (stage == 'search'):
            raise _BudgetAbort('search_work_exhausted' if stage == 'search' else 'post_search_work_exhausted')
        if self.pixels + pixels > self.policy.max_pixel_work:
            raise _BudgetAbort('search_work_exhausted' if stage == 'search' else 'post_search_work_exhausted')
        self.evaluations += int(evaluation)
        self.pixels += pixels


def _sample(texture, coords):
    """Native periodic bilinear texels, with texel centres at (i+.5)/side."""
    h, w = texture.shape[:2]
    position = np.remainder(coords, 1) * (w, h) - 0.5
    base = np.floor(position).astype(np.int64)
    frac = position - base
    x, y = base[:, 0] % w, base[:, 1] % h
    fx, fy = frac[:, 0, None], frac[:, 1, None]
    return ((1 - fx) * (1 - fy) * texture[y, x]
            + fx * (1 - fy) * texture[y, (x + 1) % w]
            + (1 - fx) * fy * texture[(y + 1) % h, x]
            + fx * fy * texture[(y + 1) % h, (x + 1) % w])


def _exposure(model, observed, policy):
    """Exact separable box-constrained convex quadratic, including every edge."""
    gains, biases = [], []
    for c in range(3):
        x, y = model[:, c], observed[:, c]
        vx = float(np.mean((x - x.mean()) ** 2))
        g = float(np.mean((x - x.mean()) * (y - y.mean())) / vx) if vx > 1e-15 else 1.0
        b = float(y.mean() - g * x.mean())
        trials = []
        if policy.gain_bounds[0] <= g <= policy.gain_bounds[1] and policy.bias_bounds[0] <= b <= policy.bias_bounds[1]:
            trials.append((g, b))
        for g in policy.gain_bounds:
            trials.append((g, float(np.clip(np.mean(y - g * x), *policy.bias_bounds))))
        for b in policy.bias_bounds:
            g = float(np.dot(x, y - b) / np.dot(x, x)) if np.dot(x, x) > 1e-15 else 1.0
            trials.append((float(np.clip(g, *policy.gain_bounds)), b))
        g, b = min(trials, key=lambda pair: float(np.mean((pair[0] * x + pair[1] - y) ** 2)))
        gains.append(g)
        biases.append(b)
    gain, bias = np.asarray(gains), np.asarray(biases)
    residual = model * gain + bias - observed
    return float(np.sqrt(np.mean(residual ** 2))), tuple(gains), tuple(biases)


def _validation(texture, transform, coords, observed, candidate, policy):
    xy = coords * candidate.distance + candidate.phase
    predicted = _sample(texture, xy @ transform.T + 0.5)
    error = (predicted * candidate.gain + candidate.bias - observed) ** 2
    # Spatial held-out tiles independently reject an occlusion/material minority.
    tiles = np.floor(xy).astype(int)
    groups = np.unique(tiles, axis=0)
    local = [float(np.sqrt(np.mean(error[np.all(tiles == tile, axis=1)])))
             for tile in groups if np.count_nonzero(np.all(tiles == tile, axis=1)) >= 6]
    return float(np.sqrt(np.mean(error))), max(local, default=float(np.sqrt(np.mean(error))))


def _circular_enclosure(intervals):
    """Start and width of minimal circular arc enclosing every closed interval."""
    ordered = sorted(intervals)
    merged = []
    for lo, hi in ordered:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    gaps = [(merged[i+1][0] - hi, merged[i+1][0])
            for i, (_, hi) in enumerate(merged[:-1])]
    gaps.append((merged[0][0] + 1 - merged[-1][1], merged[0][0]))
    gap, start = max(gaps)
    return float(start % 1), float(1-gap)


def _representative_geometry(distance, phases):
    """Strict midpoint accuracy, conditional on truth belonging to the cover.

    For [l,u], worst relative-to-truth error of arithmetic midpoint is
    (u-l)/(2*l). Thus strict <5% requires u-l <.10*l. Circular covering arc
    midpoints have worst circular error width/2, requiring width <.2.
    Outward comparison makes equality and adjacent uncertain floats fail closed.
    """
    lo, hi = distance
    if (not all(math.isfinite(v) for v in distance) or not 0 < lo <= hi
            or np.nextafter(hi-lo, math.inf) >= np.nextafter(.1*lo, -math.inf)
            or any(not math.isfinite(start) or not math.isfinite(width)
                   or not 0 <= start < 1 or not 0 <= width < .2
                   or np.nextafter(width, math.inf) >= np.nextafter(.2, -math.inf)
                   for start, width in phases)):
        return None
    midpoint = float((lo+hi)/2)
    # The rounded midpoint can land exactly on the 5% boundary even when the
    # ideal midpoint is just below it. Check its two endpoint errors as well.
    if any(np.nextafter(abs(midpoint-end), math.inf) >= np.nextafter(.05*end, -math.inf)
           for end in (lo, hi)):
        return None
    # At most eight operations on phase magnitudes <=2; reserve sixteen eps
    # for midpoint/modulo endpoint error, in addition to the outward width test.
    if any(width + 16*np.finfo(float).eps >= .2 for _, width in phases):
        return None
    return midpoint, tuple(float((start+width/2) % 1) for start, width in phases)


def _box_lower(texture, transform, coords, observed, box, policy, derivatives=None):
    """Conservative interval bound; does not use fitted exposure as a bound.

    For channel c the global periodic bilinear derivative bounds are native
    side*max adjacent difference, including last-to-first. Affine d,q,phase
    bounds give a texture excursion about the centre. Intersect its range with
    [0,1]. Nonnegative bounded gain/bias then give a per-pixel predicted range.
    Distance of observed pixels to these ranges is a lower RMSE bound even when
    allowing different exposures per pixel (a relaxation of shared exposure).

    Floating guard: accepted rays have incidence >=.15 and orthonormal axes,
    hence |q|<7; d<=64, side<=128, |g|<=2, |b|<=.25, N<=2048. Each affine,
    bilinear and interval expression has fewer than 128 scalar operations on
    magnitudes <=2**18. The 1e-7 range guard exceeds 128*eps*2**18 by >10x;
    the RMS reduction guard 1e-7 exceeds gamma(8192)*16 by >1000x. This covers
    conventional IEEE double roundoff; not non-IEEE/fast-math environments.
    """
    centre = np.mean(box, axis=1)
    half = (box[:, 1] - box[:, 0]) / 2
    uv = (coords * centre[0] + centre[1:]) @ transform.T + .5
    model = _sample(texture, uv)
    lu, lv = derivatives if derivatives is not None else _derivative_bounds(texture)
    radii = np.abs(coords @ transform.T) * half[0] + np.abs(transform) @ half[1:]
    excursion = radii[:, 0, None] * lu + radii[:, 1, None] * lv + 1e-7
    lo = np.maximum(0, model - excursion)
    hi = np.minimum(1, model + excursion)
    lo = policy.gain_bounds[0] * lo + policy.bias_bounds[0] - 1e-7
    hi = policy.gain_bounds[1] * hi + policy.bias_bounds[1] + 1e-7
    difference = np.maximum(np.maximum(lo-observed, observed-hi), 0)
    return max(0., float(np.sqrt(np.mean(difference ** 2))) - 1e-7), centre, model


def estimate_unit_face(image, context, support, catalogue, fixed_policy=None):
    """Search the complete bounded single-plane hypothesis domain or abstain.

    The full distance interval times [0,1]^2 is a root for every compatible raw
    corner/texture geometry. Binary children exactly cover each closed parent.
    Only a conservative lower-bound rejection removes a box; all terminal
    boxes remain until final incumbent filtering. A cap never certifies coverage.
    Held-out bytes are read after generation search and never choose its rank,
    parameters, branch order, photometry, bounds or uncertainty enclosures.
    """
    policy = fixed_policy or SearchPolicy()
    evidence = dict(context_sha256='', support_sha256='', catalogue_sha256='',
                    source_generation_sha256='', source_validation_sha256='',
                    policy_sha256=policy.sha256 if type(policy) is SearchPolicy else '')
    budget = _WorkBudget(policy) if type(policy) is SearchPolicy else None
    abort_reason = ''
    complete = False
    candidates = ()
    representative = None
    cover = ()
    retained = 0
    conditioning = None
    lower = None
    separation = None
    admission_threshold = None
    compatible_count = unique_count = native_bytes = estimated_pixels = 0

    def charge(pixels, *, evaluation=False, stage='search'):
        nonlocal abort_reason
        try:
            budget.charge(pixels, evaluation=evaluation, stage=stage)
            return True
        except _BudgetAbort as error:
            abort_reason = error.reason
            return False

    def result(reason, distance=None, phases=None):
        resource = reason in ('catalogue_faces_cap', 'compatible_faces_cap', 'unique_template_cap',
            'native_rgba_bytes_cap', 'root_work_preflight', 'estimated_pixel_work_cap',
            'source_work_cap', 'search_work_exhausted', 'post_search_work_exhausted')
        diagnostics = tuple(replace(c, retained_geometry=None) for c in candidates) if resource else candidates
        return FitResult('abstained' if reason else 'supported', reason, diagnostics,
            None if resource else representative, () if resource else cover,
            None if resource else distance, None if resource else phases,
            None if resource else lower, None if resource else separation,
            None if resource else conditioning, budget.evaluations if budget else 0,
            policy.max_work if budget else 0, complete and not resource,
            0 if resource else retained, None if resource else admission_threshold,
            **evidence, compatible_faces=compatible_count, unique_templates=unique_count,
            native_rgba_bytes=native_bytes, estimated_pixel_work=estimated_pixels,
            actual_pixel_work=budget.pixels if budget else 0,
            pixel_work_budget=policy.max_pixel_work if budget else 0,
            domain_search_complete=complete, search_policy=policy if budget else None)

    if type(policy) is not SearchPolicy or type(context) is not PlaneContext:
        return result('invalid_estimator_input')
    evidence['context_sha256'] = context.sha256
    if type(support) is not PartitionedSupport or support.status != 'complete':
        return result('source_support_unavailable')
    # Re-run all original-witness joins and tap-role invariants before observing.
    try:
        # Reconstruct proof records too: tests may bypass frozen records, and
        # PartitionedSupport alone trusts already-created WitnessProof values.
        partition = SourceTapPartition(**asdict(support.partition))
        proofs = tuple(WitnessProof(**asdict(w)) for w in support.witnesses)
        support = PartitionedSupport('complete', '', partition, support.samples,
                                     support.groups, proofs)
    except (UnitFaceContractError, TypeError, ValueError, IndexError):
        return result('source_support_unavailable')
    evidence['support_sha256'] = support.identity_sha256
    if (support.partition.source_shape != context.source_shape
            or support.partition.conditioning_sha256 != context.conditioning_sha256):
        return result('upstream_conditioning_mismatch')
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.shape != context.source_shape + (3,)):
        return result('invalid_source_image')
    if type(catalogue) is not UnitFaceCatalogue or not catalogue.complete_within_subset:
        return result('asset_catalogue_unavailable')
    evidence['catalogue_sha256'] = catalogue.sha256
    if len(catalogue.faces) > policy.max_catalogue_faces:
        return result('catalogue_faces_cap')
    if not catalogue.faces:
        return result('unsupported_assets' if catalogue.exclusions else 'missing_assets')
    for indices in (support.generation_pixels, support.validation_pixels):
        if len(indices) < policy.min_pixels_per_role:
            return result('insufficient_source_evidence')
        if len(indices) > policy.max_pixels_per_role:
            return result('source_work_cap')
    try:
        coords, side, incidence = _rays(context, support.generation_pixels)
        vcoords, vside, vincidence = _rays(context, support.validation_pixels)
    except (ValueError, UnitFaceContractError):
        return result('grazing_or_mixed_plane')
    if incidence < policy.min_incidence or vincidence < policy.min_incidence or vside != side:
        return result('grazing_or_mixed_plane')
    slot = {0: ('west', 'east'), 1: ('down', 'up'), 2: ('north', 'south')}[context.axis][side < 0]
    compatible_count = sum(f.face_slot == slot for f in catalogue.faces)
    if not compatible_count:
        return result('unsupported_assets')
    if compatible_count > policy.max_compatible_faces:
        return result('compatible_faces_cap')
    geometries = {}
    exact_keys = {}
    native = {}
    native_digests = {}
    try:
        for face in catalogue.faces:
            if face.face_slot != slot:
                continue
            native_key = (face.texture_size, face.rgba)
            if native_key not in native:
                native_bytes += len(face.rgba)
                if native_bytes > policy.max_native_rgba_bytes:
                    return result('native_rgba_bytes_cap')
                native[native_key] = face
            transform, geometry = _face_geometry(face, context.axis, native_digests.get(native_key))
            native_digests[native_key] = face.rgba_sha256
            exact_key = (native_key, tuple(tuple(v for v in row) for row in transform), face.face_slot)
            if geometry in exact_keys and exact_keys[geometry] != exact_key:
                return result('geometry_digest_collision')
            identity = face.block_id + '[' + ','.join(k+'='+v for k, v in face.properties) + ']|' + face.sha256
            if geometry not in geometries:
                exact_keys[geometry] = exact_key
                unique_count += 1
                if unique_count > policy.max_unique_templates:
                    return result('unique_template_cap')
                geometries[geometry] = [face, native_key, transform, []]
            geometries[geometry][3].append(identity)
            previous = geometries[geometry][0]
            previous_id = previous.block_id + '[' + ','.join(k+'='+v for k, v in previous.properties) + ']|' + previous.sha256
            if identity < previous_id:
                geometries[geometry][0] = face
    except (ValueError, TypeError):
        return result('unsupported_assets')
    ng, nv = len(support.generation_pixels), len(support.validation_pixels)
    if unique_count + 1 > policy.max_work:
        return result('root_work_preflight')
    estimated_pixels = (policy.max_work+7)*ng + (unique_count+1)*nv
    if estimated_pixels > policy.max_pixel_work:
        return result('estimated_pixel_work_cap')
    # If even the nearest admissible plane cannot resolve any candidate's
    # finest texels, no source measurement/search can establish this anchor.
    # Coordinates consume no image values; generation split only at preflight.
    h, w = context.source_shape
    if any(i % w == w-1 or i//w == h-1 for i in support.generation_pixels + support.validation_pixels):
        return result('border_source_evidence')
    try:
        qx, _, _ = _rays(context, tuple(i+1 for i in support.generation_pixels))
        qy, _, _ = _rays(context, tuple(i+w for i in support.generation_pixels))
    except ValueError:
        return result('grazing_or_mixed_plane')
    nearest_footprint = policy.distance_bounds[0] * max(
        float(np.max(np.linalg.norm(qx-coords, axis=1))),
        float(np.max(np.linalg.norm(qy-coords, axis=1))))
    if all(nearest_footprint*key[0][0] > 1/policy.min_pixels_per_texel
           for _, key, _, _ in geometries.values()):
        return result('insufficient_resolvable_texels')
    # All structural/ray/work checks precede reading even generation RGB and
    # float texture allocation. Cache one raw native texture and its derivatives
    # across exact state aliases and distinct raw-corner orientations.
    native = {key: (_native_texture(face), None) for key, face in native.items()}
    native = {key: (texture, _derivative_bounds(texture)) for key, (texture, _) in native.items()}
    derivative_cache = {}
    for geometry, group in geometries.items():
        texture, derivatives = native[group[1]]
        group[1] = texture
        group[3] = tuple(sorted(group[3]))
        derivative_cache[geometry] = derivatives
    generation = observe_role_pixels(image, support, 'generation')
    evidence['source_generation_sha256'] = generation.identity_sha256
    observed = np.frombuffer(generation.rgb, np.uint8).reshape(-1, 3).astype(float)/255
    if np.max(np.std(observed, axis=0)) < .01:
        return result('flat_or_uninformative_evidence')
    # Finest source-ray footprint is measured per split after fitting, below.
    best = {g: None for g in geometries}
    queue = []
    leaves = []
    serial = 0
    incumbent = math.inf
    removed_floor = math.inf

    def evaluate(geometry, box):
        nonlocal serial, incumbent
        if not charge(ng, evaluation=True):
            return False
        face, texture, transform, aliases = geometries[geometry]
        lb, centre, model = _box_lower(texture, transform, coords, observed, box, policy,
                                     derivative_cache[geometry])
        error, gain, bias = _exposure(model, observed, policy)
        item = Candidate(face.sha256, geometry, tuple(sorted(aliases)), float(centre[0]),
                         tuple(float(v % 1) for v in centre[1:]), gain, bias, error, None, None)
        if best[geometry] is None or error < best[geometry].generation_rmse:
            best[geometry] = item
        incumbent = min(incumbent, error)
        serial += 1
        heapq.heappush(queue, (lb, serial, geometry, box))
        return True

    root = np.asarray((policy.distance_bounds, (0., 1.), (0., 1.)))
    for geometry in sorted(geometries):
        if not evaluate(geometry, root.copy()):
            return result(abort_reason)
    while queue:
        lb, _, geometry, box = heapq.heappop(queue)
        threshold = min(incumbent + 1e-7 + policy.residual_slack, policy.generation_rmse_limit)
        if lb > threshold:
            removed_floor = min(removed_floor, lb)
            continue
        widths = box[:, 1]-box[:, 0]
        limits = np.asarray((policy.leaf_relative_distance * box[0, 0],
                             policy.leaf_phase, policy.leaf_phase))
        if np.all(widths <= limits):
            leaves.append((lb, geometry, box))
            continue
        dimension = int(np.argmax(widths / limits))
        midpoint = float(np.mean(box[dimension]))
        children = (box.copy(), box.copy())
        children[0][dimension, 1] = midpoint
        children[1][dimension, 0] = midpoint
        for child in children:
            if not evaluate(geometry, child):
                candidates = tuple(sorted((c for c in best.values() if c is not None),
                                          key=lambda c: (c.generation_rmse, c.geometry_sha256)))
                return result(abort_reason)
    complete = True
    threshold = min(incumbent + 1e-7 + policy.residual_slack, policy.generation_rmse_limit)
    admission_threshold = threshold
    removed_floor = min(removed_floor, min((lb for lb, _, _ in leaves if lb > threshold),
                                          default=math.inf))
    leaves = [(lb, g, b) for lb, g, b in leaves if lb <= threshold]
    cover = tuple(ParameterBox(g, tuple(float(v) for v in b[0]),
        tuple(float(v) for v in b[1]), tuple(float(v) for v in b[2]), float(lb))
        for lb, g, b in leaves)
    candidates = tuple(sorted(best.values(), key=lambda c: (c.generation_rmse, c.geometry_sha256)))
    retained = len(leaves)
    lower = min((lb for lb, _, _ in leaves), default=removed_floor)
    if not math.isfinite(lower):
        lower = None
    if not leaves or incumbent > policy.generation_rmse_limit:
        return result('generation_residual_failure')
    live = {g for _, g, _ in leaves}
    candidates = tuple(replace(c, retained_geometry=c.geometry_sha256 in live) for c in candidates)
    if len(live) != 1:
        return result('competing_geometry_aliases')
    winner = candidates[0]
    if winner.geometry_sha256 not in live:
        return result('generation_residual_failure')
    distance = (float(min(b[0, 0] for _, _, b in leaves)),
                float(max(b[0, 1] for _, _, b in leaves)))
    phases = tuple(_circular_enclosure([(float(b[a, 0]), float(b[a, 1]))
                                       for _, _, b in leaves]) for a in (1, 2))
    if (distance[0] <= policy.distance_bounds[0] or distance[1] >= policy.distance_bounds[1]):
        return result('distance_search_boundary', distance, phases)
    geometry_point = _representative_geometry(distance, phases)
    if geometry_point is None:
        return result('scale_or_phase_ambiguous', distance, phases)
    separation = max(0., removed_floor - incumbent) if math.isfinite(removed_floor) else None
    # Measurement conditioning: RGB prediction Jacobian, generation-only,
    # normalized distance column to fractional distance; no validation refit.
    _, texture, transform, _ = geometries[winner.geometry_sha256]
    representative_distance, representative_phase = geometry_point
    if not charge(ng, evaluation=True, stage='representative'):
        return result(abort_reason)
    model = _sample(texture, (coords*representative_distance + representative_phase) @ transform.T + .5)
    error, gain, bias = _exposure(model, observed, policy)
    representative = Candidate(winner.face_sha256, winner.geometry_sha256, winner.state_aliases,
        representative_distance, representative_phase, gain, bias, error, None, None, True)
    # Keep the original best objective and its admission band. Midpoint residual
    # never becomes a looser incumbent that could re-admit a previously pruned box.
    if error + 1e-7 > policy.generation_rmse_limit:
        return result('representative_generation_residual_failure', distance, phases)
    if not charge(ng, stage='spatial'):
        return result(abort_reason)
    _, generation_tile_error = _validation(texture, transform, coords, observed, representative, policy)
    if generation_tile_error > 2*policy.generation_rmse_limit:
        return result('generation_spatial_residual_failure', distance, phases)
    params = np.asarray((representative.distance, *representative.phase))
    derivatives = []
    for axis in range(3):
        step = 1e-5 * representative.distance if axis == 0 else 1e-5
        plus, minus = params.copy(), params.copy()
        plus[axis] += step
        minus[axis] -= step
        if not charge(ng, stage='derivatives'):
            return result(abort_reason)
        a = _sample(texture, (coords * plus[0] + plus[1:]) @ transform.T + .5)
        if not charge(ng, stage='derivatives'):
            return result(abort_reason)
        b = _sample(texture, (coords * minus[0] + minus[1:]) @ transform.T + .5)
        derivatives.append(((a-b) * representative.gain / (2*step) * (representative.distance if axis == 0 else 1)).ravel())
    singular = np.linalg.svd(np.asarray(derivatives).T, compute_uv=False)
    conditioning = float(singular[-1]/singular[0]) if singular[0] > 0 else 0.
    if conditioning < .05:
        return result('ill_conditioned_geometry', distance, phases)
    # Read held-out pixels only now. Rank and all generation parameters frozen.
    validation = observe_role_pixels(image, support, 'validation')
    evidence['source_validation_sha256'] = validation.identity_sha256
    # Source resolution check in both independent splits, from actual neighboring
    # source rays, not a rectified density or an assumed sixteen-texel texture.
    for observation in (generation, validation):
        indices = observation.source_indices
        h, w = context.source_shape
        if any(i % w == w-1 or i//w == h-1 for i in indices):
            return result('border_source_evidence', distance, phases)
        q, _, _ = _rays(context, indices)
        qx, _, _ = _rays(context, tuple(i+1 for i in indices))
        qy, _, _ = _rays(context, tuple(i+w for i in indices))
        footprint = max(float(np.max(np.linalg.norm(qx-q, axis=1))),
                        float(np.max(np.linalg.norm(qy-q, axis=1))))
        if distance[1] * footprint * texture.shape[0] > 1/policy.min_pixels_per_texel:
            return result('insufficient_resolvable_texels', distance, phases)
    observed_validation = np.frombuffer(validation.rgb, np.uint8).reshape(-1, 3)/255
    if not charge(nv, stage='validation'):
        return result(abort_reason)
    error, local = _validation(texture, transform, vcoords, observed_validation, representative, policy)
    representative = replace(representative, validation_rmse=error, validation_max_tile_rmse=local)
    measured = []
    for candidate in candidates:
        _, t, matrix, _ = geometries[candidate.geometry_sha256]
        if not charge(nv, stage='validation'):
            return result(abort_reason)
        error, local = _validation(t, matrix, vcoords, observed_validation, candidate, policy)
        measured.append(replace(candidate, validation_rmse=error, validation_max_tile_rmse=local))
    candidates = tuple(measured)  # identical generation rank, including all rivals
    if (representative.validation_rmse + 1e-7 > policy.validation_rmse_limit
            or representative.validation_max_tile_rmse + 1e-7 > 2*policy.validation_rmse_limit):
        return result('validation_residual_failure', distance, phases)
    return result('', distance, phases)

