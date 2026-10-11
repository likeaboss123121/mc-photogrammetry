"""Pure, image-only VGGT camera/depth boundary; no legacy solver promotion."""
from dataclasses import dataclass
import math

import numpy as np

from mc_photogrammetry.pipeline.contracts import CameraIntrinsics
from mc_photogrammetry.reconstruction.camera import CameraModel, camera_rotation_matrix


@dataclass(frozen=True)
class PreprocessingTransform:
    """Pillow resize, crop, then pad; integer coordinates denote pixel centres."""

    source_size: tuple[int, int]
    resized_size: tuple[int, int]
    crop_box: tuple[int, int, int, int]
    padding: tuple[int, int, int, int] = (0, 0, 0, 0)  # left, top, right, bottom

    def __post_init__(self):
        w, h = self.source_size
        rw, rh = self.resized_size
        left, top, right, bottom = self.crop_box
        if min(w, h, rw, rh) <= 0 or min(self.padding) < 0:
            raise ValueError("invalid image dimensions/padding")
        if not (0 <= left < right <= rw and 0 <= top < bottom <= rh):
            raise ValueError("crop must lie inside resized image")

    @property
    def model_size(self):
        l, t, r, b = self.crop_box
        pl, pt, pr, pb = self.padding
        return r - l + pl + pr, b - t + pt + pb

    @property
    def affine(self):
        sx, sy = np.asarray(self.resized_size) / np.asarray(self.source_size)
        l, t, _, _ = self.crop_box
        pl, pt, _, _ = self.padding
        return np.array([[sx, 0, (sx - 1) / 2 - l + pl],
                         [0, sy, (sy - 1) / 2 - t + pt], [0, 0, 1]], dtype=float)

    def to_model(self, pixels):
        return map_pixels(pixels, self.affine)

    def to_source(self, pixels):
        return map_pixels(pixels, np.linalg.inv(self.affine))

    def valid_source_mask(self):
        w, h = self.model_size
        l, t, r, b = self.crop_box
        pl, pt, _, _ = self.padding
        yy, xx = np.indices((h, w))
        return (xx >= pl) & (xx < pl + r - l) & (yy >= pt) & (yy < pt + b - t)

    def to_dict(self):
        return {"source_size": self.source_size, "resized_size": self.resized_size,
                "crop_box": self.crop_box, "padding_ltrb": self.padding,
                "model_size": self.model_size, "source_to_model_affine": self.affine.tolist(),
                "pixel_centres": "integer; Pillow half-pixel resize mapping"}


def vggt_transforms(source_sizes, mode="crop"):
    """Exact dimension/crop/batch-padding decisions from pinned upstream load_fn."""
    if not source_sizes or mode not in ("crop", "pad"):
        raise ValueError("nonempty source sizes and crop/pad required")
    interim = []
    for w, h in source_sizes:
        if min(w, h) <= 0:
            raise ValueError("source size must be positive")
        if mode == "pad" and h > w:
            rh, rw = 518, round(w * 518 / h / 14) * 14
        else:
            rw, rh = 518, round(h * 518 / w / 14) * 14
        if min(rw, rh) <= 0:
            raise ValueError("upstream resize rounds a dimension to zero")
        top = (rh - 518) // 2 if mode == "crop" and rh > 518 else 0
        ch = min(rh, 518) if mode == "crop" else rh
        cw = rw
        if mode == "pad":
            pl, pt = (518 - cw) // 2, (518 - ch) // 2
            padding = (pl, pt, 518 - cw - pl, 518 - ch - pt)
        else:
            padding = (0, 0, 0, 0)
        interim.append(PreprocessingTransform((w, h), (rw, rh), (0, top, cw, top + ch), padding))
    mw = max(x.model_size[0] for x in interim)
    mh = max(x.model_size[1] for x in interim)
    result = []
    for x in interim:
        w, h = x.model_size
        pl, pt, pr, pb = x.padding
        left, top = (mw - w) // 2, (mh - h) // 2
        result.append(PreprocessingTransform(x.source_size, x.resized_size, x.crop_box,
                      (pl + left, pt + top, pr + mw - w - left, pb + mh - h - top)))
    return result


def map_pixels(pixels, affine):
    p = np.asarray(pixels, dtype=float)
    if p.shape[-1] != 2 or not np.isfinite(p).all():
        raise ValueError("finite pixel pairs required")
    hom = np.concatenate((p, np.ones((*p.shape[:-1], 1))), axis=-1)
    mapped = hom @ np.asarray(affine).T
    return mapped[..., :2] / mapped[..., 2:3]


def quaternion_rotation_error_bound(unit_roundoff):
    """Frobenius envelope for pinned VGGT quat_to_mat, normal finite arithmetic.

    Norm sum: gamma_4; reciprocal/division: a=(gamma_4+u)/(1-gamma_4).
    Quadratic pair: gamma_2; final multiply: u. A diagonal's absolute
    error is at most eta=2*b+u*(1+2*b), where b bounds those products.
    Nine entries imply ||R_raw-R_exact||_F <= 3*eta. No lost C1 data used.
    """
    u = float(unit_roundoff)
    gamma4, gamma2 = 4 * u / (1 - 4 * u), 2 * u / (1 - 2 * u)
    a = (gamma4 + u) / (1 - gamma4)
    b = a + gamma2 + u + a * gamma2 + a * u + gamma2 * u + a * gamma2 * u
    return 3 * (2 * b + u * (1 + 2 * b))


FLOAT64_UNIT_ROUNDOFF = np.finfo(np.float64).eps / 2
CANONICAL_ROTATION_RESIDUAL_BOUND = 64 * np.finfo(np.float64).eps


def rotation_diagnostics(extrinsic):
    raw = np.asarray(extrinsic)
    result = {"dtype": str(raw.dtype), "shape": list(raw.shape)}
    if raw.shape != (3, 4) or not np.isfinite(raw).all():
        result["finite"] = False
        return result
    r = raw[:, :3].astype(np.float64)
    result.update(finite=True, unavailable={})
    operations = {"determinant": lambda: np.linalg.det(r),
                  "orthogonality_frobenius": lambda: np.linalg.norm(r @ r.T - np.eye(3)),
                  "rotation_condition": lambda: np.linalg.cond(r)}
    for name, operation in operations.items():
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                value = float(operation())
            if not math.isfinite(value):
                raise ValueError("nonfinite diagnostic")
            result[name] = value
        except Exception as exc:
            result[name] = None
            result["unavailable"][name] = f"{type(exc).__name__}: {exc}"
    return result


@dataclass(frozen=True)
class CanonicalPose:
    raw_extrinsic: np.ndarray
    raw_intrinsic: np.ndarray
    extrinsic: np.ndarray
    intrinsic: np.ndarray
    diagnostics: dict


class PosePolicyRefusal(ValueError):
    def __init__(self, reason, diagnostics):
        super().__init__(reason)
        self.diagnostics = diagnostics


def canonicalize_pose(extrinsic, intrinsic):
    """Explicit one-time polar correction; keep raw predictions unchanged."""
    raw_e, raw_k = np.asarray(extrinsic).copy(), np.asarray(intrinsic).copy()
    if raw_e.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("declared float32 or float64 pose precision required")
    e, k = _validate_pinhole(raw_e, raw_k)
    r = e[:, :3]
    diagnostics = rotation_diagnostics(raw_e)
    determinant = diagnostics["determinant"]
    if determinant is None or determinant <= 0:
        raise PosePolicyRefusal("proper finite rotation determinant required", diagnostics)
    try:
        left, singular, right = np.linalg.svd(r)
    except Exception as exc:
        diagnostics["canonicalization_unavailable"] = f"{type(exc).__name__}: {exc}"
        raise PosePolicyRefusal("rotation SVD unavailable", diagnostics) from exc
    canonical_r = left @ right
    bound = quaternion_rotation_error_bound(np.finfo(raw_e.dtype).eps / 2)
    correction = float(np.linalg.norm(canonical_r - r))
    if not math.isfinite(correction) or not np.isfinite(singular).all():
        diagnostics["canonicalization_unavailable"] = "nonfinite correction/SVD result"
        raise PosePolicyRefusal("nonfinite rotation correction", diagnostics)
    diagnostics.update(correction_frobenius=correction, correction_bound_frobenius=bound,
                       singular_values=singular.tolist())
    # Fail closed at the derived input envelope. Do not add an empirical SVD
    # correction budget; validate its float64 orthogonality separately.
    if singular[-1] <= 0 or correction > bound:
        raise PosePolicyRefusal("non-rigid or over-correction pose refused", diagnostics)
    canonical_e = np.column_stack((canonical_r, e[:, 3]))
    validate_camera(canonical_e, k)
    diagnostics.update(policy="one float64 polar/SO(3) correction; translation unchanged",
                       assumed_construction="VGGT quaternion formula; finite normal intermediates",
                       correction_frobenius=correction, correction_bound_frobenius=bound,
                       canonical_residual_tolerance=CANONICAL_ROTATION_RESIDUAL_BOUND,
                       canonical_orthogonality_frobenius=float(np.linalg.norm(canonical_r @ canonical_r.T - np.eye(3))),
                       canonical_determinant=float(np.linalg.det(canonical_r)),
                       singular_values=singular.tolist())
    for value in (raw_e, raw_k, canonical_e, k):
        value.setflags(write=False)
    return CanonicalPose(raw_e, raw_k, canonical_e, k, diagnostics)


def _validate_pinhole(extrinsic, intrinsic):
    e, k = np.asarray(extrinsic, dtype=float), np.asarray(intrinsic, dtype=float)
    if e.shape != (3, 4) or k.shape != (3, 3) or not np.isfinite(e).all() or not np.isfinite(k).all():
        raise ValueError("finite 3x4 extrinsic and 3x3 intrinsic required")
    if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.array_equal(k[2], [0, 0, 1]) or k[0, 1] != 0 or k[1, 0] != 0:
        raise ValueError("positive axis-aligned pinhole intrinsics required")
    return e, k


def validate_camera(extrinsic, intrinsic):
    """Geometry accepts only a canonical float64 rotation, never raw drift."""
    e, k = _validate_pinhole(extrinsic, intrinsic)
    r = e[:, :3]
    if np.linalg.det(r) <= 0:
        raise ValueError("OpenCV extrinsic requires a proper rotation")
    if np.linalg.norm(r @ r.T - np.eye(3)) > CANONICAL_ROTATION_RESIDUAL_BOUND:
        raise ValueError("raw near-rotation requires explicit canonicalize_pose")
    return e, k


def validate_depth_confidence(depth, confidence, valid_source):
    d, c, m = np.asarray(depth), np.asarray(confidence), np.asarray(valid_source)
    if d.shape != c.shape or d.shape != m.shape or m.dtype != np.bool_:
        raise ValueError("depth/confidence/source mask shapes must agree")
    if not np.isfinite(d[m]).all() or (d[m] <= 0).any():
        raise ValueError("observed source requires finite positive axial depth")
    if not np.isfinite(c[m]).all() or (c[m] < 0).any():
        raise ValueError("observed source requires finite nonnegative confidence")
    return m.copy()  # confidence is an uncalibrated proposal weight, not probability


def unproject_axial(pixels, depth, extrinsic, intrinsic):
    e, k = validate_camera(extrinsic, intrinsic)
    p = np.asarray(pixels, dtype=float)
    d = np.asarray(depth, dtype=float)
    if d.shape != p.shape[:-1] or not np.isfinite(d).all() or (d <= 0).any():
        raise ValueError("finite positive axial depths must match pixels")
    if not np.isfinite(p).all():
        raise ValueError("finite pixels required")
    rays = np.concatenate(((p - k[:2, 2]) / np.diag(k)[:2], np.ones((*p.shape[:-1], 1))), axis=-1)
    return (rays * d[..., None] - e[:, 3]) @ e[:, :3]


def project_world(points, extrinsic, intrinsic):
    e, k = validate_camera(extrinsic, intrinsic)
    p = np.asarray(points, dtype=float)
    camera = p @ e[:, :3].T + e[:, 3]
    if not np.isfinite(camera).all() or (camera[..., 2] <= 0).any():
        raise ValueError("points must be finite and in front of camera")
    hom = camera @ k.T
    return hom[..., :2] / hom[..., 2:3], camera[..., 2]


def source_intrinsic(model_intrinsic, transform):
    return np.linalg.inv(transform.affine) @ np.asarray(model_intrinsic)


def repository_camera(extrinsic, intrinsic, image_size):
    """Keep model-world gauge, convert y-down camera rows to repository y-up."""
    e, k = validate_camera(extrinsic, intrinsic)
    rotation = np.diag([1, -1, 1]) @ e[:, :3]
    forward = rotation[2]
    horizontal = math.hypot(forward[0], forward[2])
    pitch = math.degrees(math.atan2(-forward[1], horizontal))
    yaw = math.degrees(math.atan2(-forward[0], forward[2])) if horizontal > 8 * np.finfo(float).eps else 0.0
    centre = -(e[:, :3].T @ e[:, 3])
    base = CameraModel(tuple(centre), yaw, pitch, eye_height=0)
    residual = rotation @ camera_rotation_matrix(base).T
    roll = math.degrees(math.atan2(residual[1, 0], residual[0, 0]))
    result = CameraModel(tuple(centre), yaw, pitch, eye_height=0, roll_degrees=roll,
                         intrinsics=CameraIntrinsics(tuple(image_size), (k[0, 0], k[1, 1]), (k[0, 2], k[1, 2])))
    if np.linalg.norm(camera_rotation_matrix(result) - rotation) > CANONICAL_ROTATION_RESIDUAL_BOUND:
        raise ValueError("camera angle conversion lost rotation")
    return result


def projection_delta_bound(camera_points, point_displacement_bound, intrinsic):
    """Float64 evaluation of a perspective lemma, not an outward certificate.

    Does not cover rounded inputs or cancellation in homogeneous projection.
    Used only to form diagnostic estimates, never an acceptance threshold.
    """
    v = np.asarray(camera_points, dtype=float)
    d = np.broadcast_to(np.asarray(point_displacement_bound, dtype=float), v.shape[:-1])
    if not np.isfinite(v).all() or not np.isfinite(d).all() or (d < 0).any() or (v[..., 2] <= d).any():
        raise ValueError("ill-conditioned/nonpositive camera-depth envelope")
    z = v[..., 2]
    axis_bound = np.diag(intrinsic)[:2] * d[..., None] * (z[..., None] + np.abs(v[..., :2])) / (z * (z - d))[..., None]
    if not np.isfinite(axis_bound).all():
        raise ValueError("ill-conditioned perspective bound")
    return np.linalg.norm(axis_bound, axis=-1)


def canonical_roundtrip_roundoff_estimate(pixels, depth, pose):
    """Partial model-pixel roundoff estimate; NOT a certified error bound.

    Omits rounded inverse affine, source K construction/application, homogeneous
    cancellation, repository centre/subtraction and Euler conversion arithmetic.
    Measured SO(3) residual is not an outward rounding certificate either.
    """
    e, k = pose.extrinsic, pose.intrinsic
    p, d = np.asarray(pixels, dtype=float), np.asarray(depth, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or d.shape != (len(p),) or not np.isfinite(p).all() or not np.isfinite(d).all() or (d <= 0).any():
        raise ValueError("finite pixels and positive depths required for conditioning")
    view = np.column_stack(((p - k[:2, 2]) / np.diag(k)[:2] * d[:, None], d))
    delta = view - e[:, 3]
    world = delta @ e[:, :3]
    u = FLOAT64_UNIT_ROUNDOFF
    gamma3, gamma4 = 3 * u / (1 - 3 * u), 4 * u / (1 - 4 * u)
    # Pixel subtraction/division/depth multiply; subtraction of translation;
    # three-term inverse and forward dot products plus addition of translation.
    view_error = gamma3 * np.abs(view)
    inverse_error = (view_error + u * (np.abs(view) + np.abs(e[:, 3]))) @ np.abs(e[:, :3])
    inverse_error += gamma3 * np.abs(delta) @ np.abs(e[:, :3])
    camera_error = inverse_error @ np.abs(e[:, :3]).T
    camera_error += gamma4 * (np.abs(world) @ np.abs(e[:, :3]).T + np.abs(e[:, 3]))
    camera_error += np.linalg.norm(e[:, :3] @ e[:, :3].T - np.eye(3)) * np.linalg.norm(delta, axis=1)[:, None]
    displacement = np.linalg.norm(camera_error, axis=1)
    bound = projection_delta_bound(view, displacement, k)
    # Projection multiply/divide and principal-point addition.
    return bound + gamma4 * (np.linalg.norm(p - k[:2, 2], axis=1) + np.linalg.norm(k[:2, 2]))


@dataclass(frozen=True)
class ScaleUncertainty:
    metric_units_per_model_unit: float | None = None
    blocks_per_model_unit: float | None = None
    block_scale_alternatives: tuple[float, ...] = ()
    orientation_status: str = "model_gauge_unresolved"

    def __post_init__(self):
        scales = (*self.block_scale_alternatives, self.metric_units_per_model_unit, self.blocks_per_model_unit)
        if any(x is not None and (not math.isfinite(x) or x <= 0) for x in scales):
            raise ValueError("scale hypotheses must be finite and positive")
        if self.blocks_per_model_unit is not None and self.block_scale_alternatives:
            raise ValueError("selected block scale cannot coexist with unresolved alternatives")

    def to_dict(self):
        return {"metric_units_per_model_unit": self.metric_units_per_model_unit,
                "blocks_per_model_unit": self.blocks_per_model_unit,
                "block_scale_alternatives": self.block_scale_alternatives,
                "orientation_status": self.orientation_status,
                "accuracy": "unknown", "confidence_semantics": "uncalibrated proposal weight"}
