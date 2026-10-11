"""V2-B split-phase calibration harness (evaluation-only, prereg 2026-09-14).

This module is the evaluation-only implementation of
``docs/reports/M4-V2B-split-phase-calibration-prereg-2026-09-14.md`` (SHA-256
``8D76846C9CAE20136461123116EC71280530BE6E449FD2DDA928BB55AC7E8449``).  It
owns the frozen 1000-tuple grid literal and its validator, the deterministic
RNG-free closed-form 96-control / 864-observation generator with its sealed
labels, the exact ``Fraction`` ``x_i``/``B``/``I``/``accept`` classifier, the
first-feasible lexicographic scan, and the corpus/sample/control/provenance/
patch disjointness re-checks.

Boundaries (prereg sections 1, 5, 7, 10): this module performs **no** file,
network, process, environment, or dynamic-import IO.  It may import
``lattice_planes`` read-only (P0) and uses only the standard library plus
``numpy``.  It never reads a source image, camera manifest, F3 payload, ground
truth, mask, or texture label; every control is synthetic and declared before
synthesis.  ``predict_corpus`` accepts no label path/handle and never touches
``SplitPhaseCalibration``; it calls the frozen P0 producer once per observation,
requires ``status == "complete"`` and an exact replay, consumes only the bound
local-axis ``D`` profile, and retains/validates the non-owning profile.  Only
the score path consumes labels, and only after the prediction bytes, digest, and
``frozen_at`` exist.

Determinism: every scheduled value is dyadic or an exactly convertible finite
binary64, so patches, geometries, labels, and digests regenerate
byte-identically with no RNG and no caller seed.
"""
from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from . import lattice
from . import lattice_planes as _lp

# --- Frozen version / domain literals ---------------------------------------

SPLIT_PHASE_CALIBRATION_CORPUS_VERSION = "v2b-p0-calibration-corpus-1"
SPLIT_PHASE_CALIBRATION_PREDICTIONS_VERSION = "v2b-p0-calibration-predictions-1"
SPLIT_PHASE_CALIBRATION_AUDIT_VERSION = "v2b-p0-calibration-score-audit-1"
SPLIT_PHASE_CALIBRATION_SPLITS = ("calibration", "evaluation")
SPLIT_PHASE_CALIBRATION_SIGMA = {"calibration": 0, "evaluation": 1}
SPLIT_PHASE_CALIBRATION_CLASSES = (
    "textureless", "same_block", "seam_null", "positive")
SPLIT_PHASE_CALIBRATION_CLASS_RANGES = {
    "textureless": (0, 16), "same_block": (16, 32),
    "seam_null": (32, 48), "positive": (48, 96),
}
SPLIT_PHASE_CALIBRATION_NULL_CLASSES = ("textureless", "same_block", "seam_null")
SPLIT_PHASE_CALIBRATION_PATCH_SHAPE = (64, 64)
SPLIT_PHASE_GRID_VERSION = "v2b-p0-calibration-grid-1"
SPLIT_PHASE_CALIBRATION_VERSION = "v2b-p0-calibration-1"
SPLIT_PHASE_PATCH_DOMAIN = b"mc-photogrammetry-split-patch-v1"
SPLIT_PHASE_LABEL_DOMAIN = "mc-photogrammetry-split-phase-label-v1"

# --- Frozen generator literals (prereg section 7) ---------------------------

SPLIT_PHASE_TAU_BAND_SET: tuple[Fraction, ...] = (
    Fraction(0), Fraction(1, 8), Fraction(1, 4), Fraction(3, 8), Fraction(1, 2),
)
SPLIT_PHASE_TAU_MARGIN_SET: tuple[Fraction, ...] = (
    Fraction(1, 8), Fraction(1, 4), Fraction(3, 8), Fraction(1, 2), Fraction(1),
)
SPLIT_PHASE_TAU_ENERGY_SET: tuple[Fraction, ...] = (
    Fraction(0), Fraction(1, 8), Fraction(1, 4), Fraction(3, 8), Fraction(1, 2),
)
SPLIT_PHASE_MIN_HELD_OUT_SET: tuple[int, ...] = (1, 2, 4, 8)
SPLIT_PHASE_MIN_DISTINCT_FRAMES_SET: tuple[int, ...] = (2, 3)
SPLIT_PHASE_TARGET_POWER = Fraction(1, 2)
SPLIT_PHASE_SELECTED_B_SET: tuple[float, ...] = (0.5, 1.0, 2.0)
SPLIT_PHASE_PERIOD_PIXELS_SET: tuple[float, ...] = (8.0, 16.0, 32.0)
SPLIT_PHASE_DELTA_MAGNITUDE = Fraction(1, 32)
SPLIT_PHASE_BOUNDARY_HALF_WIDTH = Fraction(1, 16)

# --- Frozen budgets (prereg section 7; ceilings; exact limit succeeds) ------

MAX_CONTROLS_PER_SPLIT = 96
MAX_AXES_PER_CONTROL = 3
MAX_FRAMES_PER_AXIS = 3
MAX_OBSERVATIONS_PER_SPLIT = 864
MAX_PRODUCER_CALLS_PER_SPLIT = 864
MAX_PRODUCED_PROFILE_BINS_PER_RESULT = 128
MAX_PROFILE_BINS_PER_OBSERVATION = 64
MAX_P0_REPLAY_CHECKS_PER_SPLIT = 6912
MAX_GRID_TUPLES = 1000
MAX_HYPOTHESES_PER_AXIS = 4
MAX_HYPOTHESES_TOTAL = 12
MAX_CLASSIFIER_TERMS_PER_SPLIT = 442_368_000
MAX_PATCH_BYTES_PER_SPLIT = 28_311_552
MAX_PAYLOAD_BYTES_PER_SPLIT = 2 ** 24
MAX_LABEL_BYTES_PER_SPLIT = 2 ** 24

BUDGET_CAPS: dict[str, int] = {
    "max_controls_per_split": MAX_CONTROLS_PER_SPLIT,
    "max_axes_per_control": MAX_AXES_PER_CONTROL,
    "max_frames_per_axis": MAX_FRAMES_PER_AXIS,
    "max_observations_per_split": MAX_OBSERVATIONS_PER_SPLIT,
    "max_producer_calls_per_split": MAX_PRODUCER_CALLS_PER_SPLIT,
    "max_produced_profile_bins_per_result": MAX_PRODUCED_PROFILE_BINS_PER_RESULT,
    "max_profile_bins_per_observation": MAX_PROFILE_BINS_PER_OBSERVATION,
    "max_p0_replay_checks_per_split": MAX_P0_REPLAY_CHECKS_PER_SPLIT,
    "max_grid_tuples": MAX_GRID_TUPLES,
    "max_hypotheses_per_axis": MAX_HYPOTHESES_PER_AXIS,
    "max_hypotheses_total": MAX_HYPOTHESES_TOTAL,
    "max_classifier_terms_per_split": MAX_CLASSIFIER_TERMS_PER_SPLIT,
    "max_patch_bytes_per_split": MAX_PATCH_BYTES_PER_SPLIT,
    "max_payload_bytes_per_split": MAX_PAYLOAD_BYTES_PER_SPLIT,
    "max_label_bytes_per_split": MAX_LABEL_BYTES_PER_SPLIT,
}
"""Frozen ceiling names; a caller may only lower a cap, never raise one."""


# --- Canonical JSON, digests, canonical dyadics -----------------------------

def canonical_json(value) -> bytes:
    """Frozen canonical JSON encoding used for every calibration digest."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def digest_json(value) -> str:
    """Lowercase SHA-256 over the canonical JSON of ``value``."""
    return hashlib.sha256(canonical_json(value)).hexdigest()


def digest_record(record: dict, digest_field: str) -> str:
    """Digest over a record's field dict with only its own digest removed."""
    return digest_json({key: value for key, value in record.items()
                        if key != digest_field})


def canonical_record_bytes(record) -> bytes:
    """Canonical bytes of any harness or P0 record exposing ``to_record``."""
    if hasattr(record, "to_record"):
        record = record.to_record()
    return canonical_json(record)


def _hex64(value) -> bool:
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def is_reduced_dyadic(value, *, allow_zero: bool = True, allow_one: bool = False,
                      require_positive: bool = False) -> bool:
    """True for a canonical reduced dyadic ``[numerator, denominator]``."""
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or any(type(part) is not int for part in value)):
        return False
    numerator, denominator = value
    if denominator <= 0 or (denominator & (denominator - 1)) != 0:
        return False
    if numerator == 0:
        return denominator == 1 and allow_zero and not require_positive
    if numerator < 0 or numerator > denominator:
        return False
    if numerator == denominator:
        return allow_one
    return numerator % 2 == 1


def _dyadic_pair(value) -> tuple[int, int]:
    if not is_reduced_dyadic(value, allow_zero=True, allow_one=True):
        raise ValueError("noncanonical_dyadic")
    return (int(value[0]), int(value[1]))


def dyadic(value) -> list[int]:
    """Canonical ``[numerator, denominator]`` for an exact ``Fraction``."""
    value = Fraction(value)
    return [value.numerator, value.denominator]


def dyadic_fraction(value) -> Fraction:
    """Exact ``Fraction`` of a canonical reduced dyadic pair or list."""
    if not is_reduced_dyadic(value, allow_zero=True, allow_one=True):
        raise ValueError("noncanonical_dyadic")
    return Fraction(int(value[0]), int(value[1]))


def _mod_one(value: Fraction) -> Fraction:
    return value - (value.numerator // value.denominator)


def d1_on_torus(first, second) -> Fraction:
    """Exact circular metric ``min(|x - y|, 1 - |x - y|)`` on ``[0, 1)``."""
    delta = abs(Fraction(first) - Fraction(second))
    return min(delta, 1 - delta)


def d_torus(first, second) -> Fraction:
    """Exact L-infinity torus metric over two 3-vectors of exact rationals."""
    if len(first) != 3 or len(second) != 3:
        raise ValueError("split_not_reconstructible")
    return max(d1_on_torus(Fraction(a), Fraction(b))
               for a, b in zip(first, second))


# --- Frozen generator (deterministic, RNG-free) -----------------------------

def class_of_ordinal(ordinal: int) -> str:
    """Frozen class derived from the fixed ordinal ranges (D2)."""
    for name, (start, stop) in SPLIT_PHASE_CALIBRATION_CLASS_RANGES.items():
        if start <= ordinal < stop:
            return name
    raise ValueError("split_not_reconstructible")


def local_axis(k: int, w: int, j: int) -> str:
    """Frozen bound local axis ``a(k, w, j)``."""
    return "u" if (k + w + j) % 2 == 0 else "v"


def world_sign(k: int, w: int) -> int:
    """Frozen balanced world sign ``s(k, w)``."""
    return +1 if (k + w) % 2 == 0 else -1


def selected_b(k: int, w: int, sigma: int) -> float:
    """Frozen ``selected_b`` binary64 value from ``B_SET``."""
    return SPLIT_PHASE_SELECTED_B_SET[(k + w + sigma) % 3]


def period_pixels(k: int, w: int, sigma: int) -> float:
    """Frozen ``period_pixels`` binary64 value from ``P_SET``."""
    return SPLIT_PHASE_PERIOD_PIXELS_SET[(k + w + sigma) % 3]


def alpha_true(k: int, w: int, sigma: int) -> Fraction:
    """Frozen sealed seam truth ``alpha(k, w, sigma)`` on ``[0, 1)``."""
    return Fraction((2 * k + 3 * w + sigma) % 16, 16)


def leading_cell(k: int, w: int, j: int, sigma: int) -> Fraction:
    """Frozen exact leading cell ``L(k, w, j, sigma)`` on ``[0, 1)``."""
    return Fraction((2 * k + 5 * w + 3 * j + sigma) % 16, 16)


def delta_true(k: int, w: int, j: int, sigma: int, control_class: str) -> Fraction:
    """Frozen signed generator-internal ``delta`` (zero on every null)."""
    if control_class in SPLIT_PHASE_CALIBRATION_NULL_CLASSES:
        return Fraction(0)
    if sigma == 0 and (k - 48) % 3 == 0:
        return Fraction(0)
    return (SPLIT_PHASE_DELTA_MAGNITUDE if (k + w + j + sigma) % 2 == 0
            else -SPLIT_PHASE_DELTA_MAGNITUDE)


def ell_true_of(k: int, w: int, j: int, sigma: int, control_class: str) -> Fraction:
    """Frozen ``ell_true = mod(L + delta, 1)``."""
    return _mod_one(leading_cell(k, w, j, sigma)
                    + delta_true(k, w, j, sigma, control_class))


def frame_translation(k: int, w: int, j: int, sigma: int,
                      control_class: str) -> Fraction:
    """Frozen ``T = mod(alpha - s*(L + delta), 1)``."""
    anchor = alpha_true(k, w, sigma)
    sign = world_sign(k, w)
    return _mod_one(anchor - sign * (leading_cell(k, w, j, sigma)
                                     + delta_true(k, w, j, sigma, control_class)))


def hypothesis_set(k: int, w: int, sigma: int) -> tuple[Fraction, ...]:
    """Frozen declared truth set ``H_w`` (``alpha`` first, then ``d`` shifts)."""
    anchor = alpha_true(k, w, sigma)
    return tuple(_mod_one(anchor + Fraction(d, 16)) for d in (0, 3, 7, 13))


def boundary_contrast(k: int, w: int) -> float:
    """Frozen ``boundary_contrast`` in ``{1.0, 1.25, 1.5, 1.75}``."""
    return float(1 + Fraction((k + w) % 4, 4))


def nuisance_amplitude(k: int, w: int) -> float:
    """Frozen ``nuisance_amplitude`` in ``{0.5, 0.75, 1.0, 1.25}``."""
    return float(Fraction(1, 2) + Fraction((k + 2 * w) % 4, 4))


def control_id(sigma: int, k: int) -> str:
    """Frozen ``control_id`` hash of ``{sigma, split, class, ordinal}``."""
    return digest_json({
        "sigma": sigma, "split": SPLIT_PHASE_CALIBRATION_SPLITS[sigma],
        "class": class_of_ordinal(k), "ordinal": k})


def observation_id(sigma: int, k: int, w: int, j: int) -> str:
    """Frozen ``observation_id`` hash of the identity preimage."""
    return digest_json({
        "sigma": sigma, "split": SPLIT_PHASE_CALIBRATION_SPLITS[sigma],
        "control_id": control_id(sigma, k), "world_axis": w, "frame": j,
        "local_axis": local_axis(k, w, j)})


def geometry_for(k: int, w: int, sigma: int) -> _lp.SplitPhaseGeometry:
    """Frozen ``SplitPhaseGeometry`` for one control axis and split."""
    return _lp.SplitPhaseGeometry(
        centre=(31.5, 31.5), extent=31.5, samples=64,
        rectified_pixel_step=lattice.rectified_pixel_step(31.5, 64),
        period_pixels=period_pixels(k, w, sigma))


def label_payload_sha256(control: str, control_class: str,
                         alpha_components: tuple | None) -> str:
    """Sealed per-control label digest bound into every one of its observations."""
    return digest_json({
        "domain": SPLIT_PHASE_LABEL_DOMAIN,
        "control_id": control,
        "class": control_class,
        "components": (None if alpha_components is None
                       else [list(value) for value in alpha_components]),
    })


def patch_sha256(patch) -> str:
    """Frozen patch framing digest over the exact little-endian bytes."""
    array = np.asarray(patch, dtype="<f8")
    rows, columns = int(array.shape[0]), int(array.shape[1])
    raw = array.tobytes(order="C")
    payload = (SPLIT_PHASE_PATCH_DOMAIN
               + rows.to_bytes(8, "little", signed=False)
               + columns.to_bytes(8, "little", signed=False)
               + len(raw).to_bytes(8, "little", signed=False) + raw)
    return hashlib.sha256(payload).hexdigest()


def profile_payload_sha256(samples_raw: bytes, weights_raw: bytes,
                           indices_raw: bytes) -> str:
    """Frozen ``payload_sha256`` framing (samples, weights, indices order)."""
    parts = [_lp.SPLIT_SEAM_PROFILE_PAYLOAD_VERSION]
    for raw in (samples_raw, weights_raw, indices_raw):
        parts.append(len(raw).to_bytes(8, "little", signed=False))
        parts.append(raw)
    return hashlib.sha256(b"".join(parts)).hexdigest()


def profile_sha256(samples, weights, indices) -> str:
    """``payload_sha256`` for ordered arrays in canonical little-endian form."""
    return profile_payload_sha256(
        np.asarray(samples, dtype="<f8").tobytes(order="C"),
        np.asarray(weights, dtype="<f8").tobytes(order="C"),
        np.asarray(indices, dtype="<i8").tobytes(order="C"))


def _patch_array_sha256_bytes(patch) -> str:
    array = np.ascontiguousarray(np.asarray(patch, dtype="<f8"))
    return patch_sha256(array)


def patch_for(k: int, w: int, j: int, sigma: int, control_class: str) -> np.ndarray:
    """Frozen class-specific patch ``P`` of shape ``(64, 64)``.

    The patch position index ``t`` is also the profile-bin index, so the exact
    residue is ``x_t = (L + selected_b * t / period_pixels) mod 1`` (the frozen
    ``selected_b/period_pixels = 1/16``), giving the exact 16-position sub-bin
    grid.  Every value is built with ``Fraction`` and materialized once into
    binary64, so the bytes are reproducible without any tolerance.
    """
    axis = local_axis(k, w, j)
    rows, columns = SPLIT_PHASE_CALIBRATION_PATCH_SHAPE
    count = rows if axis == "v" else columns
    leading = leading_cell(k, w, j, sigma)
    b_ratio = selected_b(k, w, sigma).as_integer_ratio()
    p_ratio = period_pixels(k, w, sigma).as_integer_ratio()
    step = Fraction(*b_ratio) / Fraction(*p_ratio)
    base = Fraction(*nuisance_amplitude(k, w).as_integer_ratio()) / 8 + 2 * sigma
    contrast = Fraction(*boundary_contrast(k, w).as_integer_ratio())
    nuisance = Fraction(*nuisance_amplitude(k, w).as_integer_ratio())
    ell = ell_true_of(k, w, j, sigma, control_class)
    values = np.zeros(count, dtype="<f8")
    if control_class != "textureless":
        for t in range(count):
            x_t = _mod_one(leading + step * t)
            if control_class == "same_block":
                quantity = base
            elif control_class == "seam_null":
                # q(x) == q(x + 1/2) exactly.
                quarter = 4 * x_t
                even = (quarter.numerator // quarter.denominator) % 2 == 0
                quantity = base + (nuisance if even else 0)
            else:
                quantity = (base + contrast
                            if d1_on_torus(x_t, ell) <= SPLIT_PHASE_BOUNDARY_HALF_WIDTH
                            else base)
            values[t] = float(quantity)
    patch = np.empty((rows, columns), dtype="<f8")
    if axis == "u":
        patch[:, :] = values[np.newaxis, :]
    else:
        patch[:, :] = values[:, np.newaxis]
    return patch


def calibration_producer_budget() -> _lp.SplitPhaseProducerBudget:
    """Explicit deterministic P0 budget for the fixed 64x64 synthetic patch.

    The pins are generous but finite and identical for synthesis and prediction;
    the calibration-level ceilings of prereg section 7 are enforced separately
    by :class:`CalibrationCaps` and are never widened from this budget.
    """
    return _lp._make_budget({name: 1_000_000_000 for name in _lp._SPLIT_BUDGET_FIELDS})


# --- Typed immutable observation / control / corpus records -----------------

def frame_index(observation) -> int | None:
    """Recover the frozen frame index ``j`` from the identity digest."""
    for candidate in range(MAX_FRAMES_PER_AXIS):
        if observation_id(observation.sigma, observation.ordinal,
                          observation.world_axis, candidate) \
                == observation.observation_id:
            return candidate
    return None


def observation_geometry(observation) -> _lp.SplitPhaseGeometry:
    """Rebuild the five-field frozen geometry from an observation carrier."""
    geometry = observation.geometry
    if not isinstance(geometry, dict) or set(geometry) != {
            "centre", "extent", "samples", "rectified_pixel_step", "period_pixels"}:
        raise ValueError("split_not_reconstructible")
    return _lp.SplitPhaseGeometry(
        centre=tuple(geometry["centre"]), extent=geometry["extent"],
        samples=geometry["samples"],
        rectified_pixel_step=geometry["rectified_pixel_step"],
        period_pixels=geometry["period_pixels"])


def observation_matches_frozen(observation) -> bool:
    """True only when an observation record regenerates from the frozen row."""
    j = frame_index(observation)
    if j is None:
        return False
    k, w, sigma = observation.ordinal, observation.world_axis, observation.sigma
    if observation.control_id != control_id(sigma, k):
        return False
    if observation.local_axis != local_axis(k, w, j):
        return False
    if observation.world_sign != world_sign(k, w):
        return False
    if tuple(observation.L_dyadic) != tuple(dyadic(leading_cell(k, w, j, sigma))):
        return False
    if tuple(observation.T_dyadic) != tuple(dyadic(
            frame_translation(k, w, j, sigma, observation.control_class))):
        return False
    if tuple(observation.H_w) != tuple(
            tuple(dyadic(value)) for value in hypothesis_set(k, w, sigma)):
        return False
    if observation.selected_b != selected_b(k, w, sigma):
        return False
    if tuple(observation.selected_b_ratio) != tuple(
            int(part) for part in selected_b(k, w, sigma).as_integer_ratio()):
        return False
    if observation.period_pixels != period_pixels(k, w, sigma):
        return False
    if (Fraction(*observation.boundary_contrast.as_integer_ratio())
            != Fraction(*boundary_contrast(k, w).as_integer_ratio())
            or Fraction(*observation.nuisance_amplitude.as_integer_ratio())
            != Fraction(*nuisance_amplitude(k, w).as_integer_ratio())):
        return False
    frozen_class = class_of_ordinal(k)
    if observation.control_class != frozen_class:
        return False
    expected_patch = patch_for(k, w, j, sigma, frozen_class)
    if observation.patch_sha256 != patch_sha256(expected_patch):
        return False
    try:
        decoded_patch = observation.patch()
    except ValueError:
        return False
    if decoded_patch.tobytes(order="C") != expected_patch.tobytes(order="C"):
        return False
    return observation.geometry == geometry_for(k, w, sigma).to_record()


@dataclass(frozen=True, slots=True)
class CalibrationObservation:
    """One control x world axis x frame observation (the calibration sample)."""

    observation_id: str
    sigma: int
    split: str
    control_id: str
    control_class: str
    ordinal: int
    patch_sha256: str
    L_dyadic: tuple
    T_dyadic: tuple
    alpha_dyadic_hidden: tuple
    H_w: tuple
    selected_b: float
    selected_b_ratio: tuple
    period_pixels: float
    local_axis: str
    world_axis: int
    world_sign: int
    boundary_contrast: float
    nuisance_amplitude: float
    geometry: dict
    geometry_sha256: str
    patch_float64_le_base64: str
    non_owning_profile_sha256: str
    provenance_sha256: str
    label_payload_sha256: str
    observation_sha256: str
    sealed_fields_present: bool = True

    def __post_init__(self) -> None:
        if self.sigma not in (0, 1):
            raise ValueError("split_not_reconstructible")
        if self.split != SPLIT_PHASE_CALIBRATION_SPLITS[self.sigma]:
            raise ValueError("split_not_reconstructible")
        if self.control_class not in SPLIT_PHASE_CALIBRATION_CLASSES:
            raise ValueError("split_not_reconstructible")
        if type(self.ordinal) is not int or not 0 <= self.ordinal < 96:
            raise ValueError("split_not_reconstructible")
        if self.control_class != class_of_ordinal(self.ordinal):
            raise ValueError("split_not_reconstructible")
        for name in ("L_dyadic", "T_dyadic", "alpha_dyadic_hidden"):
            _dyadic_pair(getattr(self, name))
        if not isinstance(self.H_w, tuple) or len(self.H_w) != 4:
            raise ValueError("split_not_reconstructible")
        for value in self.H_w:
            _dyadic_pair(value)
        if self.selected_b not in SPLIT_PHASE_SELECTED_B_SET:
            raise ValueError("split_not_reconstructible")
        if (not isinstance(self.selected_b_ratio, tuple)
                or len(self.selected_b_ratio) != 2
                or any(type(part) is not int for part in self.selected_b_ratio)):
            raise ValueError("split_not_reconstructible")
        if Fraction(*self.selected_b_ratio) != Fraction(
                *self.selected_b.as_integer_ratio()):
            raise ValueError("split_not_reconstructible")
        if self.period_pixels not in SPLIT_PHASE_PERIOD_PIXELS_SET:
            raise ValueError("split_not_reconstructible")
        if self.local_axis not in ("u", "v"):
            raise ValueError("split_not_reconstructible")
        if type(self.world_axis) is not int or self.world_axis not in (0, 1, 2):
            raise ValueError("split_not_reconstructible")
        if self.world_sign not in (-1, 1):
            raise ValueError("split_not_reconstructible")
        for name in ("geometry_sha256", "patch_sha256", "non_owning_profile_sha256",
                     "provenance_sha256", "label_payload_sha256",
                     "observation_sha256", "observation_id", "control_id"):
            if not _hex64(getattr(self, name)):
                raise ValueError("split_not_reconstructible")
        if (not isinstance(self.geometry, dict)
                or set(self.geometry) != {
                    "centre", "extent", "samples", "rectified_pixel_step",
                    "period_pixels"}
                or self.geometry_sha256 != digest_json(self.geometry)):
            raise ValueError("split_not_reconstructible")
        patch = self.patch()
        if patch_sha256(patch) != self.patch_sha256:
            raise ValueError("split_not_reconstructible")
        if digest_record(self.provenance_record(), "provenance_sha256") \
                != self.provenance_sha256:
            raise ValueError("split_not_reconstructible")
        if self.sealed_fields_present and (
                digest_record(self.to_record(), "observation_sha256")
                != self.observation_sha256):
            raise ValueError("split_not_reconstructible")

    def provenance_record(self) -> dict:
        """Exact ``provenance_sha256`` preimage; it carries ``sigma``."""
        return {
            "sigma": self.sigma,
            "split": self.split,
            "control_id": self.control_id,
            "observation_id": self.observation_id,
            "L_dyadic": list(self.L_dyadic),
            "T_dyadic": list(self.T_dyadic),
            "H_w": [list(value) for value in self.H_w],
            "selected_b_ratio": list(self.selected_b_ratio),
            "period_pixels": self.period_pixels,
            "a": self.local_axis,
            "w": self.world_axis,
            "s": self.world_sign,
            "geometry_sha256": self.geometry_sha256,
            "boundary_contrast": self.boundary_contrast,
            "nuisance_amplitude": self.nuisance_amplitude,
        }

    def to_record(self) -> dict:
        return {
            "observation_id": self.observation_id,
            "sigma": self.sigma,
            "split": self.split,
            "control_id": self.control_id,
            "control_class": self.control_class,
            "ordinal": self.ordinal,
            "patch_sha256": self.patch_sha256,
            "L_dyadic": list(self.L_dyadic),
            "T_dyadic": list(self.T_dyadic),
            "alpha_dyadic_hidden": list(self.alpha_dyadic_hidden),
            "H_w": [list(value) for value in self.H_w],
            "selected_b": self.selected_b,
            "selected_b_ratio": list(self.selected_b_ratio),
            "period_pixels": self.period_pixels,
            "local_axis": self.local_axis,
            "world_axis": self.world_axis,
            "world_sign": self.world_sign,
            "boundary_contrast": self.boundary_contrast,
            "nuisance_amplitude": self.nuisance_amplitude,
            "geometry": dict(self.geometry),
            "geometry_sha256": self.geometry_sha256,
            "patch_float64_le_base64": self.patch_float64_le_base64,
            "non_owning_profile_sha256": self.non_owning_profile_sha256,
            "provenance_sha256": self.provenance_sha256,
            "label_payload_sha256": self.label_payload_sha256,
            "observation_sha256": self.observation_sha256,
        }

    def redacted_record(self) -> dict:
        """Corpus view: drops only the sealed truth and its payload digest."""
        record = self.to_record()
        del record["alpha_dyadic_hidden"]
        del record["label_payload_sha256"]
        return record

    def patch(self) -> np.ndarray:
        """Decoded frozen patch ``P`` (little-endian binary64, C order)."""
        try:
            raw = base64.b64decode(self.patch_float64_le_base64, validate=True)
        except (TypeError, ValueError) as error:
            raise ValueError("split_not_reconstructible") from error
        if base64.b64encode(raw).decode("ascii") != self.patch_float64_le_base64:
            raise ValueError("split_not_reconstructible")
        rows, columns = SPLIT_PHASE_CALIBRATION_PATCH_SHAPE
        if len(raw) != rows * columns * 8:
            raise ValueError("split_not_reconstructible")
        array = np.frombuffer(raw, dtype="<f8").reshape((rows, columns)).copy()
        if not bool(np.all(np.isfinite(array))):
            raise ValueError("split_not_reconstructible")
        return array


def observation_from_redacted(record: dict, *,
                              sample_sha256: str) -> CalibrationObservation:
    """Rebuild an observation carrier from its redacted corpus record.

    The sealed ``alpha_dyadic_hidden``/``label_payload_sha256`` values are
    unrecoverable from the corpus, so the carrier is marked
    ``sealed_fields_present=False`` and keeps the frozen ``observation_sha256``
    purely as identity.  Every retained digest (patch, geometry, provenance) is
    still recomputed and checked by the constructor.
    """
    payload = dict(record)
    payload.pop("observation_sha256", None)
    payload["alpha_dyadic_hidden"] = (0, 1)
    payload["label_payload_sha256"] = digest_json({
        "domain": SPLIT_PHASE_LABEL_DOMAIN, "control_id": record["control_id"],
        "class": record["control_class"], "components": None, "redacted": True})
    payload["observation_sha256"] = sample_sha256
    payload["sealed_fields_present"] = False
    payload["H_w"] = tuple(tuple(value) for value in payload["H_w"])
    for name in ("L_dyadic", "T_dyadic", "selected_b_ratio"):
        payload[name] = tuple(payload[name])
    payload["geometry"] = dict(payload["geometry"])
    return CalibrationObservation(**payload)


@dataclass(frozen=True, slots=True)
class CalibrationControl:
    """Frozen control record: one ordinal, nine ordered observations."""

    control_id: str
    sigma: int
    split: str
    control_class: str
    ordinal: int
    observation_ids: tuple
    patch_refs: tuple
    observations: tuple
    control_sha256: str

    def __post_init__(self) -> None:
        if self.sigma not in (0, 1) or self.split != SPLIT_PHASE_CALIBRATION_SPLITS[self.sigma]:
            raise ValueError("split_not_reconstructible")
        if self.control_class != class_of_ordinal(self.ordinal):
            raise ValueError("split_not_reconstructible")
        if self.control_id != control_id(self.sigma, self.ordinal):
            raise ValueError("split_not_reconstructible")
        expected = MAX_AXES_PER_CONTROL * MAX_FRAMES_PER_AXIS
        if (not isinstance(self.observation_ids, tuple)
                or len(self.observation_ids) != expected
                or any(not _hex64(value) for value in self.observation_ids)):
            raise ValueError("split_not_reconstructible")
        if (not isinstance(self.patch_refs, tuple)
                or len(self.patch_refs) != expected
                or any(not _hex64(value) for value in self.patch_refs)):
            raise ValueError("split_not_reconstructible")
        if (not isinstance(self.observations, tuple)
                or len(self.observations) != expected
                or any(not isinstance(item, CalibrationObservation)
                       for item in self.observations)):
            raise ValueError("split_not_reconstructible")
        if any(item.control_id != self.control_id or item.sigma != self.sigma
               for item in self.observations):
            raise ValueError("split_not_reconstructible")
        if tuple(item.observation_id for item in self.observations) \
                != self.observation_ids:
            raise ValueError("split_not_reconstructible")
        if tuple(item.patch_sha256 for item in self.observations) \
                != self.patch_refs:
            raise ValueError("split_not_reconstructible")
        if (not _hex64(self.control_sha256)
                or digest_record(self.to_record(), "control_sha256")
                != self.control_sha256):
            raise ValueError("split_not_reconstructible")

    def to_record(self) -> dict:
        return {
            "control_id": self.control_id,
            "sigma": self.sigma,
            "split": self.split,
            "control_class": self.control_class,
            "ordinal": self.ordinal,
            "observation_ids": list(self.observation_ids),
            "patch_refs": list(self.patch_refs),
            "control_sha256": self.control_sha256,
        }

    def corpus_record(self) -> dict:
        record = self.to_record()
        record["observations"] = [item.redacted_record()
                                  for item in self.observations]
        return record


def make_control(k: int, sigma: int, observations: tuple) -> CalibrationControl:
    """Build the validated control record for one ordinal."""
    payload = {
        "control_id": control_id(sigma, k),
        "sigma": sigma,
        "split": SPLIT_PHASE_CALIBRATION_SPLITS[sigma],
        "control_class": class_of_ordinal(k),
        "ordinal": k,
        "observation_ids": tuple(item.observation_id for item in observations),
        "patch_refs": tuple(item.patch_sha256 for item in observations),
        "observations": observations,
    }
    record = {key: value for key, value in payload.items()
              if key != "observations"}
    return CalibrationControl(**payload,
                              control_sha256=digest_record(record, "control_sha256"))


@dataclass(frozen=True, slots=True)
class CalibrationCorpus:
    """Ordered corpus record; the only corpus object passed to predict."""

    corpus_version: str
    evaluation_only: bool
    sigma: int
    split: str
    patch_shape: tuple
    controls: tuple
    null_control_ids: tuple
    seam_control_ids: tuple
    sample_sha256s: tuple
    labels_sha256: str
    corpus_sha256: str

    def __post_init__(self) -> None:
        if (self.corpus_version != SPLIT_PHASE_CALIBRATION_CORPUS_VERSION
                or self.evaluation_only is not True
                or self.sigma not in (0, 1)
                or self.split != SPLIT_PHASE_CALIBRATION_SPLITS[self.sigma]
                or tuple(self.patch_shape) != SPLIT_PHASE_CALIBRATION_PATCH_SHAPE):
            raise ValueError("split_not_reconstructible")
        if (not isinstance(self.controls, tuple) or not self.controls
                or len(self.controls) > MAX_CONTROLS_PER_SPLIT
                or any(not isinstance(item, CalibrationControl)
                       for item in self.controls)):
            raise ValueError("split_not_reconstructible")
        if any(item.sigma != self.sigma for item in self.controls):
            raise ValueError("split_not_reconstructible")
        if [item.ordinal for item in self.controls] != sorted(
                item.ordinal for item in self.controls):
            raise ValueError("split_not_reconstructible")
        if len({item.ordinal for item in self.controls}) != len(self.controls):
            raise ValueError("split_not_reconstructible")
        observations = self.observations
        if len(self.controls) == MAX_CONTROLS_PER_SPLIT \
                and len(observations) != MAX_OBSERVATIONS_PER_SPLIT:
            raise ValueError("split_not_reconstructible")
        if (not isinstance(self.sample_sha256s, tuple)
                or len(self.sample_sha256s) != len(observations)
                or any(not _hex64(value) for value in self.sample_sha256s)
                or len(set(self.sample_sha256s)) != len(self.sample_sha256s)):
            raise ValueError("split_not_reconstructible")
        if tuple(self.sample_sha256s) != tuple(
                item.observation_sha256 for item in observations):
            raise ValueError("split_not_reconstructible")
        expected_null = tuple(item.control_id for item in self.controls
                              if item.control_class != "positive")
        expected_seam = tuple(item.control_id for item in self.controls
                              if item.control_class == "positive")
        if tuple(self.null_control_ids) != expected_null \
                or tuple(self.seam_control_ids) != expected_seam:
            raise ValueError("split_not_reconstructible")
        if set(self.null_control_ids) & set(self.seam_control_ids):
            raise ValueError("split_not_reconstructible")
        if any(not _hex64(value) for value in self.null_control_ids
               + self.seam_control_ids):
            raise ValueError("split_not_reconstructible")
        if not _hex64(self.labels_sha256):
            raise ValueError("split_not_reconstructible")
        if (not _hex64(self.corpus_sha256)
                or digest_record(self.to_record(), "corpus_sha256")
                != self.corpus_sha256):
            raise ValueError("split_not_reconstructible")

    @property
    def observations(self) -> tuple:
        return tuple(item for control in self.controls
                     for item in control.observations)

    def to_record(self) -> dict:
        return {
            "corpus_version": self.corpus_version,
            "evaluation_only": self.evaluation_only,
            "sigma": self.sigma,
            "split": self.split,
            "patch_shape": list(self.patch_shape),
            "controls": [item.corpus_record() for item in self.controls],
            "null_control_ids": list(self.null_control_ids),
            "seam_control_ids": list(self.seam_control_ids),
            "sample_sha256s": list(self.sample_sha256s),
            "labels_sha256": self.labels_sha256,
            "corpus_sha256": self.corpus_sha256,
        }

    def corpus_payload(self) -> dict:
        record = self.to_record()
        del record["corpus_sha256"]
        return record


def make_corpus(controls: tuple, sigma: int, labels_sha256: str) -> CalibrationCorpus:
    """Reseal an ordered control tuple into a validated corpus."""
    null_ids = tuple(item.control_id for item in controls
                     if item.control_class != "positive")
    seam_ids = tuple(item.control_id for item in controls
                     if item.control_class == "positive")
    payload = {
        "corpus_version": SPLIT_PHASE_CALIBRATION_CORPUS_VERSION,
        "evaluation_only": True,
        "sigma": sigma,
        "split": SPLIT_PHASE_CALIBRATION_SPLITS[sigma],
        "patch_shape": list(SPLIT_PHASE_CALIBRATION_PATCH_SHAPE),
        "controls": [item.corpus_record() for item in controls],
        "null_control_ids": list(null_ids),
        "seam_control_ids": list(seam_ids),
        "sample_sha256s": [item.observation_sha256 for item in controls
                           for item in item.observations],
        "labels_sha256": labels_sha256,
    }
    return CalibrationCorpus(
        corpus_version=payload["corpus_version"],
        evaluation_only=True, sigma=sigma, split=payload["split"],
        patch_shape=tuple(payload["patch_shape"]), controls=tuple(controls),
        null_control_ids=null_ids, seam_control_ids=seam_ids,
        sample_sha256s=tuple(payload["sample_sha256s"]),
        labels_sha256=labels_sha256,
        corpus_sha256=digest_record(payload, "corpus_sha256"))


def corpus_from_record(record: dict) -> CalibrationCorpus:
    """Rebuild a validated corpus from its canonical payload dict."""
    if not isinstance(record, dict):
        raise ValueError("split_not_reconstructible")
    payload = dict(record)
    stored_corpus_sha256 = payload.pop("corpus_sha256", None)
    if not _hex64(stored_corpus_sha256):
        raise ValueError("split_not_reconstructible")
    sigma = payload.get("sigma")
    if sigma not in (0, 1):
        raise ValueError("split_not_reconstructible")
    hashes = list(payload["sample_sha256s"])
    cursor = 0
    controls = []
    for item in payload["controls"]:
        observations = []
        for entry in item["observations"]:
            if cursor >= len(hashes):
                raise ValueError("split_not_reconstructible")
            observations.append(observation_from_redacted(
                entry, sample_sha256=hashes[cursor]))
            cursor += 1
        control = make_control(item["ordinal"], sigma, tuple(observations))
        if control.control_sha256 != item.get("control_sha256"):
            raise ValueError("split_not_reconstructible")
        controls.append(control)
    if cursor != len(hashes):
        raise ValueError("split_not_reconstructible")
    rebuilt = make_corpus(tuple(controls), sigma, payload["labels_sha256"])
    if rebuilt.corpus_sha256 != stored_corpus_sha256:
        raise ValueError("split_not_reconstructible")
    for name in ("corpus_version", "evaluation_only", "split", "patch_shape",
                 "null_control_ids", "seam_control_ids", "sample_sha256s"):
        if name in payload and rebuilt.to_record()[name] != payload[name]:
            raise ValueError("split_not_reconstructible")
    return rebuilt


# --- Sealed labels and prediction freeze ------------------------------------

@dataclass(frozen=True, slots=True)
class CalibrationLabels:
    """Sealed label file: written once in pass 1a, opened only by score."""

    entries: tuple
    labels_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.entries, tuple) or not self.entries:
            raise ValueError("split_not_reconstructible")
        for entry in self.entries:
            if (not isinstance(entry, dict)
                    or set(entry) != {"control_id", "class", "components",
                                      "absence_reason"}
                    or not _hex64(entry["control_id"])
                    or entry["class"] not in SPLIT_PHASE_CALIBRATION_CLASSES):
                raise ValueError("split_not_reconstructible")
            if entry["class"] == "positive":
                components = entry["components"]
                if (not isinstance(components, list) or len(components) != 3
                        or any(not is_reduced_dyadic(value, allow_zero=True,
                                                     allow_one=True)
                               for value in components)
                        or entry["absence_reason"] is not None):
                    raise ValueError("split_not_reconstructible")
            elif entry["components"] is not None \
                    or entry["absence_reason"] != entry["class"]:
                raise ValueError("split_not_reconstructible")
        if (not _hex64(self.labels_sha256)
                or digest_json([dict(entry) for entry in self.entries])
                != self.labels_sha256):
            raise ValueError("split_not_reconstructible")

    def to_record(self) -> dict:
        return {"entries": [dict(entry) for entry in self.entries],
                "labels_sha256": self.labels_sha256}

    def truth(self) -> dict:
        """Sealed positive alpha 3-vectors keyed by control id."""
        truth = {}
        for entry in self.entries:
            if entry["class"] == "positive" and entry["components"] is not None:
                truth[entry["control_id"]] = tuple(
                    tuple(value) for value in entry["components"])
        return truth

    def null_ids(self) -> tuple:
        return tuple(entry["control_id"] for entry in self.entries
                     if entry["class"] != "positive")

    def seam_ids(self) -> tuple:
        return tuple(entry["control_id"] for entry in self.entries
                     if entry["class"] == "positive")


def labels_from_record(record: dict) -> CalibrationLabels:
    """Validate and load the sealed label record."""
    if not isinstance(record, dict) or set(record) != {"entries", "labels_sha256"}:
        raise ValueError("split_not_reconstructible")
    entries = tuple(dict(entry) for entry in record["entries"])
    return CalibrationLabels(entries=entries,
                             labels_sha256=record["labels_sha256"])


# --- Frozen 1000-tuple grid -------------------------------------------------

def frozen_grid_tuples() -> tuple:
    """Ascending nested iteration over the frozen 1000-tuple grid (D3)."""
    tuples = []
    for tau_band in SPLIT_PHASE_TAU_BAND_SET:
        for tau_margin in SPLIT_PHASE_TAU_MARGIN_SET:
            for tau_energy in SPLIT_PHASE_TAU_ENERGY_SET:
                for min_samples in SPLIT_PHASE_MIN_HELD_OUT_SET:
                    for min_frames in SPLIT_PHASE_MIN_DISTINCT_FRAMES_SET:
                        tuples.append((
                            (tau_band.numerator, tau_band.denominator),
                            (tau_margin.numerator, tau_margin.denominator),
                            (tau_energy.numerator, tau_energy.denominator),
                            min_samples, min_frames))
    return tuple(tuples)


def validate_grid_tuple(value, *, tau_band_zero: bool = True):
    """Validate one five-element tuple against the frozen schema."""
    if not isinstance(value, tuple) or len(value) != 5:
        raise ValueError("no_feasible_calibration")
    tau_band = _dyadic_pair(value[0])
    if not is_reduced_dyadic(tau_band, allow_zero=tau_band_zero, allow_one=True):
        raise ValueError("noncanonical_dyadic")
    tau_margin = _dyadic_pair(value[1])
    if not is_reduced_dyadic(tau_margin, allow_zero=False, allow_one=True,
                             require_positive=True):
        raise ValueError("noncanonical_dyadic")
    tau_energy = _dyadic_pair(value[2])
    if not is_reduced_dyadic(tau_energy, allow_zero=True, allow_one=True):
        raise ValueError("noncanonical_dyadic")
    if type(value[3]) is not int or value[3] < 1:
        raise ValueError("no_feasible_calibration")
    if type(value[4]) is not int or value[4] < 2:
        raise ValueError("no_feasible_calibration")
    return (tau_band, tau_margin, tau_energy, value[3], value[4])


def calibration_tuple_lt(first, second) -> bool:
    """Frozen strict lexicographic order over validated tuples."""
    for index in range(3):
        left = Fraction(*first[index])
        right = Fraction(*second[index])
        if left < right:
            return True
        if right < left:
            return False
    return (first[3], first[4]) < (second[3], second[4])


def build_grid(*, tuples=None, sigma: int = 0, target_power=None):
    """Build the frozen grid literal (or a reduced tuple set for isolated tests)."""
    if tuples is None:
        tuples = frozen_grid_tuples()
    else:
        tuples = tuple(validate_grid_tuple(value) for value in tuples)
    if (not tuples or len(tuples) > MAX_GRID_TUPLES
            or any(not calibration_tuple_lt(first, second)
                   for first, second in zip(tuples, tuples[1:]))):
        raise ValueError("split_not_reconstructible")
    if target_power is None:
        target_power = (SPLIT_PHASE_TARGET_POWER.numerator,
                        SPLIT_PHASE_TARGET_POWER.denominator)
    ids = tuple(control_id(sigma, k) for k in range(96))
    payload = {
        "grid_version": SPLIT_PHASE_GRID_VERSION,
        "evaluation_only": True,
        "target_power": list(_dyadic_pair(target_power)),
        "tuples": [[list(value[0]), list(value[1]), list(value[2]), value[3],
                    value[4]] for value in tuples],
        "null_control_ids": list(ids[:48]),
        "seam_control_ids": list(ids[48:]),
    }
    return _lp.SplitPhaseCalibrationGrid(
        grid_version=SPLIT_PHASE_GRID_VERSION, evaluation_only=True,
        target_power=tuple(payload["target_power"]), tuples=tuples,
        null_control_ids=tuple(payload["null_control_ids"]),
        seam_control_ids=tuple(payload["seam_control_ids"]),
        grid_sha256=digest_record(payload, "grid_sha256"))


FROZEN_GRID = build_grid()
"""The frozen evaluation-only 1000-tuple grid literal (prereg D3)."""
FROZEN_GRID_SHA256 = FROZEN_GRID.grid_sha256


def validate_frozen_grid(grid) -> bool:
    """Evaluator check: the grid is the frozen literal, record-for-record."""
    if not isinstance(grid, _lp.SplitPhaseCalibrationGrid):
        return False
    return canonical_record_bytes(grid) == canonical_record_bytes(FROZEN_GRID)


def validate_grid_structure(grid) -> bool:
    """Structural grid check used by the calibration-record evaluator gates.

    The exact frozen-literal equality is enforced separately by
    :func:`validate_frozen_grid` on the run path.
    """
    if not isinstance(grid, _lp.SplitPhaseCalibrationGrid):
        return False
    if grid.grid_version != SPLIT_PHASE_GRID_VERSION \
            or grid.evaluation_only is not True:
        return False
    if (tuple(grid.target_power)
            != (SPLIT_PHASE_TARGET_POWER.numerator,
                SPLIT_PHASE_TARGET_POWER.denominator)):
        return False
    if not grid.tuples or len(grid.tuples) > MAX_GRID_TUPLES:
        return False
    try:
        validated = tuple(validate_grid_tuple(value) for value in grid.tuples)
    except ValueError:
        return False
    if tuple(validated) != tuple(grid.tuples):
        return False
    if any(not calibration_tuple_lt(first, second)
           for first, second in zip(grid.tuples, grid.tuples[1:])):
        return False
    if (tuple(grid.null_control_ids)
            != tuple(control_id(0, k) for k in range(48))
            or tuple(grid.seam_control_ids)
            != tuple(control_id(0, k) for k in range(48, 96))):
        return False
    return grid.grid_sha256 == digest_record(grid.to_record(), "grid_sha256")


# --- Budgets ----------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CalibrationCaps:
    """Explicit frozen ceilings; a caller may only lower them, never raise."""

    values: tuple

    def __post_init__(self) -> None:
        if not isinstance(self.values, tuple) or len(self.values) != len(BUDGET_CAPS):
            raise ValueError("work_over_cap")
        for name, value in zip(BUDGET_CAPS, self.values):
            if type(value) is not int or value < 0 or value > BUDGET_CAPS[name]:
                raise ValueError("work_over_cap")

    def get(self, name: str) -> int:
        return self.values[tuple(BUDGET_CAPS).index(name)]

    def lower(self, **overrides: int) -> "CalibrationCaps":
        values = list(self.values)
        for name, override in overrides.items():
            if name not in BUDGET_CAPS:
                raise ValueError("work_over_cap")
            if type(override) is not int or override < 0 \
                    or override > BUDGET_CAPS[name]:
                raise ValueError("work_over_cap")
            values[tuple(BUDGET_CAPS).index(name)] = override
        return CalibrationCaps(tuple(values))

    def to_record(self) -> dict:
        return dict(zip(BUDGET_CAPS, self.values))


def default_caps() -> CalibrationCaps:
    """The frozen ceilings used by the authorized run."""
    return CalibrationCaps(tuple(BUDGET_CAPS.values()))


class _CapExceeded(Exception):
    """Internal carrier for an atomic typed cap refusal."""

    def __init__(self, cap: str):
        super().__init__(cap)
        self.cap = cap


# --- Disjointness -----------------------------------------------------------

def corpus_summary(corpus: CalibrationCorpus) -> dict:
    """Digest/set summary used for the frozen cross-split disjointness checks."""
    observations = corpus.observations
    return {
        "corpus_sha256": corpus.corpus_sha256,
        "observation_ids": [item.observation_id for item in observations],
        "sample_sha256s": [item.observation_sha256 for item in observations],
        "provenance_sha256s": [item.provenance_sha256 for item in observations],
        "nontextureless_patch_sha256s": [
            item.patch_sha256 for item in observations
            if item.control_class != "textureless"],
        "textureless": [
            {"ordinal": item.ordinal, "world_axis": item.world_axis,
             "provenance_sha256": item.provenance_sha256,
             "sample_sha256": item.observation_sha256,
             "geometry_sha256": item.geometry_sha256,
             "patch_sha256": item.patch_sha256}
            for item in observations if item.control_class == "textureless"],
    }


def _textureless_key(item: dict) -> tuple:
    return (item["ordinal"], item["world_axis"])


def disjointness_refusal(calibration: CalibrationCorpus,
                         evaluation: CalibrationCorpus) -> str | None:
    """Return the first frozen section 4 violation, or ``None``.

    Split names (``calibration``/``evaluation``) establish nothing; only the
    ``sigma``-bound digest and set checks below are evidence.
    """
    if calibration.sigma != 0 or evaluation.sigma != 1:
        return "split_leakage"
    first = corpus_summary(calibration)
    second = corpus_summary(evaluation)
    if first["corpus_sha256"] == second["corpus_sha256"]:
        return "split_leakage"
    if set(first["observation_ids"]) & set(second["observation_ids"]):
        return "split_leakage"
    if set(first["sample_sha256s"]) & set(second["sample_sha256s"]):
        return "split_leakage"
    if set(first["provenance_sha256s"]) & set(second["provenance_sha256s"]):
        return "split_leakage"
    if (set(first["nontextureless_patch_sha256s"])
            & set(second["nontextureless_patch_sha256s"])):
        return "split_leakage"
    textureless_second = {}
    for item in second["textureless"]:
        textureless_second.setdefault(_textureless_key(item), []).append(item)
    for item in first["textureless"]:
        for other in textureless_second.get(_textureless_key(item), ()):
            if item["provenance_sha256"] == other["provenance_sha256"]:
                return "split_leakage"
            if item["sample_sha256"] == other["sample_sha256"]:
                return "split_leakage"
            if item["geometry_sha256"] == other["geometry_sha256"]:
                return "split_leakage"
    return None


def validate_calibration_record_refusal(calibration: CalibrationCorpus,
                                        evaluation: CalibrationCorpus, grid,
                                        calibration_record) -> str | None:
    """Evaluator checks of prereg section 8 over a built ``SplitPhaseCalibration``."""
    if not isinstance(calibration_record, _lp.SplitPhaseCalibration):
        return "split_not_reconstructible"
    if not validate_frozen_grid(grid):
        return "split_not_reconstructible"
    if not validate_grid_structure(grid):
        return "split_not_reconstructible"
    if calibration_record.calibration_grid_sha256 != grid.grid_sha256:
        return "split_not_reconstructible"
    if calibration_record.partition_config_sha256 != _lp.SPLIT_PHASE_PARTITION_CONFIG_SHA256:
        return "split_not_reconstructible"
    selected = (tuple(calibration_record.tau_band),
                tuple(calibration_record.tau_margin),
                tuple(calibration_record.tau_energy),
                calibration_record.min_held_out_samples,
                calibration_record.min_distinct_frames_per_axis)
    if tuple(selected) not in tuple(tuple(value) for value in grid.tuples):
        return "split_not_reconstructible"
    if calibration_record.calibration_corpus_sha256 != calibration.corpus_sha256:
        return "split_not_reconstructible"
    if calibration_record.evaluation_corpus_sha256 != evaluation.corpus_sha256:
        return "split_not_reconstructible"
    if tuple(calibration_record.calibration_sample_sha256s) != calibration.sample_sha256s:
        return "split_not_reconstructible"
    if tuple(calibration_record.evaluation_sample_sha256s) != evaluation.sample_sha256s:
        return "split_not_reconstructible"
    if not calibration_record.null_control_ids \
            or not calibration_record.seam_control_ids:
        return "split_not_reconstructible"
    if tuple(calibration_record.null_control_ids) != calibration.null_control_ids:
        return "split_not_reconstructible"
    if tuple(calibration_record.seam_control_ids) != calibration.seam_control_ids:
        return "split_not_reconstructible"
    if len(calibration_record.calibration_sample_sha256s) != len(
            calibration.sample_sha256s) or not calibration.sample_sha256s:
        return "split_not_reconstructible"
    if len(calibration_record.evaluation_sample_sha256s) != len(
            evaluation.sample_sha256s) or not evaluation.sample_sha256s:
        return "split_not_reconstructible"
    if set(calibration_record.null_control_ids) & set(calibration_record.seam_control_ids):
        return "split_not_reconstructible"
    if (set(calibration_record.calibration_sample_sha256s)
            & set(calibration_record.evaluation_sample_sha256s)):
        return "split_leakage"
    return disjointness_refusal(calibration, evaluation)


# --- Exact classifier (prereg D4 + V2-B section 7) --------------------------

def classify_observation(observation: CalibrationObservation, hypothesis,
                         tau_band, samples, weights) -> dict:
    """Exact ``B``/``I``/``accept`` for one observation and hypothesis.

    ``B`` and ``I`` are coverage-normalized inclusive-band sums over the
    consumed bound profile only, indexed by ``x_i = (L + b*i/period) mod 1``
    with no half-bin offset and normalized by ``sum(C[i])``.
    """
    energies = np.asarray(samples)
    coverages = np.asarray(weights)
    if energies.ndim != 1 or coverages.shape != energies.shape:
        raise ValueError("split_not_reconstructible")
    step = Fraction(*observation.selected_b_ratio) \
        / Fraction(*observation.period_pixels.as_integer_ratio())
    leading = dyadic_fraction(observation.L_dyadic)
    ell = _mod_one(Fraction(observation.world_sign)
                   * (Fraction(hypothesis)
                      - dyadic_fraction(observation.T_dyadic)))
    interior = _mod_one(ell + Fraction(1, 2))
    tau_band = Fraction(tau_band)
    boundary_energy = Fraction(0)
    boundary_coverage = Fraction(0)
    interior_energy = Fraction(0)
    interior_coverage = Fraction(0)
    for index in range(energies.size):
        x_i = _mod_one(leading + step * index)
        energy = Fraction(*float(energies[index]).as_integer_ratio())
        coverage = Fraction(*float(coverages[index]).as_integer_ratio())
        if d1_on_torus(x_i, ell) <= tau_band:
            boundary_energy += energy
            boundary_coverage += coverage
        if d1_on_torus(x_i, interior) <= tau_band:
            interior_energy += energy
            interior_coverage += coverage
    if boundary_coverage == 0 or interior_coverage == 0:
        return {"status": "zero_coverage",
                "boundary_energy": boundary_energy,
                "interior_energy": interior_energy,
                "boundary_coverage": boundary_coverage,
                "interior_coverage": interior_coverage}
    boundary = boundary_energy / boundary_coverage
    internal = interior_energy / interior_coverage
    return {"status": "complete", "boundary_energy": boundary,
            "interior_energy": internal, "margin": boundary - internal,
            "boundary_coverage": boundary_coverage,
            "interior_coverage": interior_coverage}


def hypothesis_accepts(observation: CalibrationObservation, hypothesis,
                       tau_band, tau_margin, tau_energy, samples, weights) -> bool:
    """Exact single-hypothesis accept predicate for one observation."""
    outcome = classify_observation(observation, hypothesis, tau_band,
                                   samples, weights)
    if outcome["status"] != "complete":
        return False
    return bool(outcome["margin"] >= Fraction(tau_margin)
                and outcome["boundary_energy"] >= Fraction(tau_energy))


def classify_observation_for_tau(observation, tau_band, samples, weights) -> tuple:
    """Exact outcomes for every declared hypothesis at one ``tau_band``."""
    return tuple(classify_observation(observation, dyadic_fraction(hypothesis),
                                      tau_band, samples, weights)
                 for hypothesis in observation.H_w)


def evaluate_axis(observations, tau_band, tau_margin, tau_energy,
                  min_held_out: int, min_frames: int, *,
                  samples, weights, coverage_valid,
                  classifications: list | None = None) -> dict:
    """Determined/abstained axis status from the observations mapped to it.

    Each observation is a distinct frame.  A hypothesis is accepted for a
    qualifying observation only when ``B - I >= tau_margin`` and
    ``B >= tau_energy``; a ``zero_coverage`` observation refuses and is excluded
    from the qualifying set, as is an observation whose bound held-out coverage
    is below ``min_held_out``.  The axis is determined only when exactly one
    hypothesis is accepted by every qualifying observation and the qualifying
    frame count clears ``min_frames``.

    ``classifications`` optionally carries the precomputed per-observation
    exact ``B``/``I`` outcomes for this ``tau_band`` (they are independent of
    ``tau_margin``/``tau_energy``), so the 1000-tuple scan neither recomputes
    nor under-counts the frozen classifier terms.
    """
    intersection = None
    qualifying = 0
    flags = []
    zero_coverage_ids = []
    no_seam_evidence_ids = []
    for position, observation in enumerate(observations):
        outcomes = (classifications[position] if classifications is not None
                    else classify_observation_for_tau(
                        observation, tau_band, samples[position],
                        weights[position]))
        per_hypothesis = []
        refused = False
        for outcome in outcomes:
            if outcome["status"] != "complete":
                per_hypothesis.append(False)
                refused = True
            else:
                per_hypothesis.append(bool(
                    outcome["margin"] >= Fraction(tau_margin)
                    and outcome["boundary_energy"] >= Fraction(tau_energy)))
        flags.append(tuple(per_hypothesis))
        if refused:
            zero_coverage_ids.append(observation.observation_id)
            continue
        if coverage_valid[position] < min_held_out:
            continue
        qualifying += 1
        accepted = {index for index, ok in enumerate(per_hypothesis) if ok}
        if not accepted:
            no_seam_evidence_ids.append(observation.observation_id)
        intersection = accepted if intersection is None else (intersection & accepted)
    base = {"flags": tuple(flags),
            "zero_coverage_observation_ids": tuple(zero_coverage_ids),
            "no_seam_evidence_observation_ids": tuple(no_seam_evidence_ids),
            "qualifying_frames": qualifying}
    if intersection is None or qualifying == 0 or not intersection:
        return {**base, "status": "abstained", "reason": "no_seam_evidence",
                "emitted_alpha": None}
    if len(intersection) > 1:
        return {**base, "status": "abstained",
                "reason": "multiple_hypotheses_supported", "emitted_alpha": None}
    if qualifying < min_frames:
        return {**base, "status": "abstained", "reason": "insufficient_frames",
                "emitted_alpha": None}
    index = next(iter(intersection))
    return {**base, "status": "determined", "reason": None,
            "emitted_alpha": dyadic_fraction(observations[0].H_w[index])}


# --- Prediction (label-blind) ----------------------------------------------

def profile_arrays(evidence) -> tuple[np.ndarray, np.ndarray]:
    """Decode the canonical ``E``/``C`` little-endian arrays of one evidence."""
    if not isinstance(evidence, _lp.SplitSeamProfileEvidence):
        raise ValueError("split_not_reconstructible")
    samples = np.frombuffer(base64.b64decode(
        evidence.profile_samples_float64_le_base64, validate=True),
        dtype="<f8").copy()
    weights = np.frombuffer(base64.b64decode(
        evidence.profile_weights_float64_le_base64, validate=True),
        dtype="<f8").copy()
    if samples.size != evidence.sample_count or weights.size != evidence.weight_count:
        raise ValueError("split_not_reconstructible")
    return samples, weights


def _profile_digest(patch, geometry, local_axis: str) -> tuple[str, int]:
    """Exact frozen ``d_profile_sha256`` for one patch axis, without P0.

    Rebuilds the frozen lane partition and the ``SplitSeamProfileEvidence``
    payload/D-profile framing digest directly from the frozen patch, geometry,
    and builtin partition config, so synthesis never calls the P0 producer or
    validator.  Returns ``(d_profile_sha256, sample_count)``.  Nothing here
    scores, classifies, or reads a label; the retained non-owning digest is a
    pure function of bytes the synthesis already owns.
    """
    if local_axis not in ("u", "v"):
        raise ValueError("split_not_reconstructible")
    array = np.asarray(patch)
    if array.ndim != 2:
        raise ValueError("split_not_reconstructible")
    array = np.ascontiguousarray(array.astype("<f8", copy=False))
    height, width = int(array.shape[0]), int(array.shape[1])
    if height < 1 or width < 1:
        raise ValueError("split_not_reconstructible")
    if not bool(np.all(np.isfinite(array))) or bool(np.any(array < 0.0)):
        raise ValueError("split_not_reconstructible")
    if local_axis == "u":
        plan = _lp.split_phase_lane_plan(
            "u", height, width, _lp.SPLIT_PHASE_PARTITION_CONFIG)
    else:
        plan = _lp.split_phase_lane_plan(
            "v", width, height, _lp.SPLIT_PHASE_PARTITION_CONFIG)
    if not isinstance(plan, _lp.SplitPhaseLanePlan):
        raise ValueError("split_not_reconstructible")
    start, stop = int(plan.d_start), int(plan.d_stop)
    lane_count = stop - start
    if local_axis == "u":
        samples = array[start:stop, :].sum(axis=0)
        count = width
    else:
        samples = array[:, start:stop].sum(axis=1)
        count = height
    weights = np.full(count, lane_count, dtype="<f8")
    indices = np.arange(count, dtype="<i8")
    coverage = {
        "total": lane_count * count, "valid": lane_count * count,
        "per_bin": tuple([lane_count] * count),
    }
    payload_sha256 = profile_payload_sha256(
        np.asarray(samples, dtype="<f8").tobytes(order="C"),
        np.asarray(weights, dtype="<f8").tobytes(order="C"),
        np.asarray(indices, dtype="<i8").tobytes(order="C"))
    digest = _lp._split_d_profile_sha256(
        local_axis, float(geometry.period_pixels), count, count, count,
        coverage, payload_sha256)
    return digest, count


def _produce_observation(observation) -> "_lp.SplitPhaseProducerResult":
    patch = observation.patch()
    geometry = observation_geometry(observation)
    j = frame_index(observation)
    if j is None:
        raise ValueError("split_not_reconstructible")
    budget = calibration_producer_budget()
    result = _lp.produce_split_phase_evidence(
        patch, geometry=geometry,
        frame=f"frame-{observation.ordinal}-{observation.world_axis}-{j}",
        plane_index=observation.ordinal, reading_axis=observation.world_axis,
        source_region_id=None,
        partition_config=_lp.SPLIT_PHASE_PARTITION_CONFIG, budget=budget)
    if result.status != "complete":
        raise ValueError("split_not_reconstructible")
    if not _lp.validate_split_phase_evidence(
            result, patch, geometry=geometry,
            partition_config=_lp.SPLIT_PHASE_PARTITION_CONFIG, budget=budget):
        raise ValueError("split_not_reconstructible")
    return result


def _validate_predict_workload(corpus: CalibrationCorpus, grid, caps) -> None:
    if corpus.sigma not in (0, 1):
        raise ValueError("split_not_reconstructible")
    if len(corpus.controls) > caps.get("max_controls_per_split"):
        raise _CapExceeded("max_controls_per_split")
    if len(corpus.observations) > caps.get("max_observations_per_split"):
        raise _CapExceeded("max_observations_per_split")
    if len(corpus.observations) > caps.get("max_producer_calls_per_split"):
        raise _CapExceeded("max_producer_calls_per_split")
    if (len(corpus.observations) * 8) > caps.get("max_p0_replay_checks_per_split"):
        raise _CapExceeded("max_p0_replay_checks_per_split")
    patch_bytes = sum(
        SPLIT_PHASE_CALIBRATION_PATCH_SHAPE[0]
        * SPLIT_PHASE_CALIBRATION_PATCH_SHAPE[1] * 8
        for _ in corpus.observations)
    if patch_bytes > caps.get("max_patch_bytes_per_split"):
        raise _CapExceeded("max_patch_bytes_per_split")
    for control in corpus.controls:
        axes = {item.world_axis for item in control.observations}
        if len(axes) > caps.get("max_axes_per_control"):
            raise _CapExceeded("max_axes_per_control")
        if not axes or min(axes) < 0 or max(axes) >= MAX_AXES_PER_CONTROL:
            raise ValueError("split_not_reconstructible")
        frames = {}
        for item in control.observations:
            frames[item.world_axis] = frames.get(item.world_axis, 0) + 1
        if frames and max(frames.values()) > caps.get("max_frames_per_axis"):
            raise _CapExceeded("max_frames_per_axis")
        if any(len(item.H_w) != MAX_HYPOTHESES_PER_AXIS
               for item in control.observations):
            raise ValueError("split_not_reconstructible")
        if MAX_HYPOTHESES_PER_AXIS > caps.get("max_hypotheses_per_axis"):
            raise _CapExceeded("max_hypotheses_per_axis")
        # One shared hypothesis set per bound world axis; frames reuse it.
        hypotheses = len(axes) * MAX_HYPOTHESES_PER_AXIS
        if hypotheses > caps.get("max_hypotheses_total"):
            raise _CapExceeded("max_hypotheses_total")
    if MAX_PRODUCED_PROFILE_BINS_PER_RESULT > caps.get(
            "max_produced_profile_bins_per_result"):
        raise _CapExceeded("max_produced_profile_bins_per_result")
    if MAX_PROFILE_BINS_PER_OBSERVATION > caps.get("max_profile_bins_per_observation"):
        raise _CapExceeded("max_profile_bins_per_observation")
    if len(grid.tuples) > caps.get("max_grid_tuples"):
        raise _CapExceeded("max_grid_tuples")
    terms = (len(grid.tuples) * len(corpus.observations)
             * MAX_HYPOTHESES_PER_AXIS * MAX_PROFILE_BINS_PER_OBSERVATION * 2)
    if terms > caps.get("max_classifier_terms_per_split"):
        raise _CapExceeded("max_classifier_terms_per_split")


def _build_prediction_payload(corpus: CalibrationCorpus, grid, caps) -> dict:
    observations = corpus.observations
    tau_bands = sorted({Fraction(*value[0]) for value in grid.tuples})
    cache: dict = {}
    profile_views: dict = {}
    per_observation = []
    replay_checks = 0
    producer_calls = 0
    payload_bytes = 0
    for observation in observations:
        if not observation_matches_frozen(observation):
            raise ValueError("split_not_reconstructible")
        result = _produce_observation(observation)
        producer_calls += 1
        if producer_calls > caps.get("max_producer_calls_per_split"):
            raise _CapExceeded("max_producer_calls_per_split")
        produced_bins = sum(item.sample_count for item in result.seam_profile_evidence)
        if produced_bins > caps.get("max_produced_profile_bins_per_result"):
            raise _CapExceeded("max_produced_profile_bins_per_result")
        evidence = {item.local_axis: item for item in result.seam_profile_evidence}
        if set(evidence) != {"u", "v"}:
            raise ValueError("split_not_reconstructible")
        owning = evidence[observation.local_axis]
        non_owning = evidence["v" if observation.local_axis == "u" else "u"]
        if owning.sample_count > caps.get("max_profile_bins_per_observation"):
            raise _CapExceeded("max_profile_bins_per_observation")
        if owning.d_profile_sha256 == non_owning.d_profile_sha256:
            raise ValueError("split_not_reconstructible")
        if non_owning.d_profile_sha256 != observation.non_owning_profile_sha256:
            raise ValueError("split_not_reconstructible")
        if result.runtime_work is None:
            raise ValueError("split_not_reconstructible")
        replay_checks += result.runtime_work.replay_checks
        if replay_checks > caps.get("max_p0_replay_checks_per_split"):
            raise _CapExceeded("max_p0_replay_checks_per_split")
        non_owning_record = non_owning.to_record()
        payload_bytes += (
            len(base64.b64decode(non_owning.profile_samples_float64_le_base64))
            + len(base64.b64decode(non_owning.profile_weights_float64_le_base64))
            + len(base64.b64decode(non_owning.axis_bin_indices_int64_le_base64)))
        if payload_bytes > caps.get("max_payload_bytes_per_split"):
            raise _CapExceeded("max_payload_bytes_per_split")
        samples, weights = profile_arrays(owning)
        profile_views[observation.observation_id] = (samples, weights,
                                                     owning.coverage["valid"])
        per_band = {}
        for tau_band in tau_bands:
            per_band[tau_band] = classify_observation_for_tau(
                observation, tau_band, samples, weights)
        cache[observation.observation_id] = per_band
        per_observation.append({
            "observation_id": observation.observation_id,
            "local_axis": observation.local_axis,
            "consumed_profile_sha256": owning.d_profile_sha256,
            "non_owning_profile_sha256": non_owning.d_profile_sha256,
            "non_owning_profile": non_owning_record,
        })
    terms = (len(grid.tuples) * len(observations)
             * MAX_HYPOTHESES_PER_AXIS * MAX_PROFILE_BINS_PER_OBSERVATION * 2)
    per_control = []
    for control in corpus.controls:
        by_axis = {0: [], 1: [], 2: []}
        for observation in control.observations:
            by_axis[observation.world_axis].append(observation)
        tuple_results = []
        for value in grid.tuples:
            tau_band = Fraction(*value[0])
            tau_margin = Fraction(*value[1])
            tau_energy = Fraction(*value[2])
            axes = []
            for world_axis in (0, 1, 2):
                group = by_axis[world_axis]
                if not group:
                    raise ValueError("split_not_reconstructible")
                outcome = evaluate_axis(
                    group, tau_band, tau_margin, tau_energy, value[3], value[4],
                    samples=[profile_views[item.observation_id][0] for item in group],
                    weights=[profile_views[item.observation_id][1] for item in group],
                    coverage_valid=[profile_views[item.observation_id][2]
                                    for item in group],
                    classifications=[cache[item.observation_id][tau_band]
                                     for item in group])
                emitted = outcome["emitted_alpha"]
                axes.append({
                    "world_axis": world_axis,
                    "status": outcome["status"],
                    "reason": outcome["reason"],
                    "emitted_alpha": (None if emitted is None
                                      else dyadic(emitted)),
                    "flags": [list(flag) for flag in outcome["flags"]],
                    "zero_coverage_observation_ids":
                        list(outcome["zero_coverage_observation_ids"]),
                    "no_seam_evidence_observation_ids":
                        list(outcome["no_seam_evidence_observation_ids"]),
                    "qualifying_frames": outcome["qualifying_frames"],
                })
            tuple_results.append(axes)
        per_control.append({
            "control_id": control.control_id,
            "control_class": control.control_class,
            "classifier_term_count": terms,
            "tuple_results": tuple_results,
        })
    return {
        "freeze_version": SPLIT_PHASE_CALIBRATION_PREDICTIONS_VERSION,
        "evaluation_only": True,
        "sigma": corpus.sigma,
        "grid_sha256": grid.grid_sha256,
        "corpus_sha256": corpus.corpus_sha256,
        "labels_sha256": corpus.labels_sha256,
        "label_blind": True,
        "per_control": per_control,
        "per_observation": per_observation,
        "classifier_term_count": terms,
    }


def predict_corpus(corpus: CalibrationCorpus, grid=None, *,
                   caps: CalibrationCaps | None = None,
                   expected_grid_sha256: str | None = None) -> dict:
    """Label-blind scoring of a corpus under a grid.

    Returns a typed ``complete`` outcome carrying the prediction payload to be
    resealed, or an atomic typed ``refused`` outcome.  It never receives and
    never reads a label path, handle, or content.
    """
    if caps is None:
        caps = default_caps()
    if grid is None:
        grid = FROZEN_GRID
    if expected_grid_sha256 is not None and grid.grid_sha256 != expected_grid_sha256:
        return {"status": "refused", "reason": "split_not_reconstructible"}
    try:
        _validate_predict_workload(corpus, grid, caps)
        payload = _build_prediction_payload(corpus, grid, caps)
    except _CapExceeded as error:
        return {"status": "refused", "reason": "work_over_cap",
                "cap": error.cap}
    except ValueError as error:
        return {"status": "refused", "reason": str(error)}
    return {"status": "complete", "payload": payload}


def freeze_prediction_payload(payload: dict, *, frozen_at: str) -> dict:
    """Attach ``predictions_sha256`` then ``frozen_at`` (digest precedes time)."""
    body = dict(payload)
    body.pop("predictions_sha256", None)
    body.pop("frozen_at", None)
    sealed = dict(body)
    sealed["predictions_sha256"] = digest_record(body, "predictions_sha256")
    sealed["frozen_at"] = frozen_at
    return sealed


def verify_prediction_digest(prediction: dict) -> bool:
    """True only when ``predictions_sha256`` matches the frozen body."""
    if not isinstance(prediction, dict):
        return False
    digest = prediction.get("predictions_sha256")
    if not _hex64(digest):
        return False
    body = {key: value for key, value in prediction.items()
            if key not in ("predictions_sha256", "frozen_at")}
    return digest_record(body, "predictions_sha256") == digest


def prediction_refusal(prediction: dict, corpus: CalibrationCorpus, grid) -> str | None:
    """Evaluator checks of prereg section 8 over a frozen prediction record."""
    if not isinstance(prediction, dict) or not verify_prediction_digest(prediction):
        return "split_not_reconstructible"
    if (prediction.get("freeze_version")
            != SPLIT_PHASE_CALIBRATION_PREDICTIONS_VERSION
            or prediction.get("evaluation_only") is not True
            or prediction.get("label_blind") is not True):
        return "split_not_reconstructible"
    if prediction.get("sigma") != corpus.sigma:
        return "split_not_reconstructible"
    if prediction.get("grid_sha256") != grid.grid_sha256:
        return "split_not_reconstructible"
    if prediction.get("corpus_sha256") != corpus.corpus_sha256:
        return "split_not_reconstructible"
    if prediction.get("labels_sha256") != corpus.labels_sha256:
        return "split_not_reconstructible"
    controls = prediction.get("per_control")
    observations = prediction.get("per_observation")
    if not isinstance(controls, list) or not isinstance(observations, list):
        return "split_not_reconstructible"
    if [item["control_id"] for item in controls] \
            != [item.control_id for item in corpus.controls]:
        return "split_not_reconstructible"
    if [item["observation_id"] for item in observations] \
            != [item.observation_id for item in corpus.observations]:
        return "split_not_reconstructible"
    terms = (len(grid.tuples) * len(corpus.observations)
             * MAX_HYPOTHESES_PER_AXIS * MAX_PROFILE_BINS_PER_OBSERVATION * 2)
    if prediction.get("classifier_term_count") != terms:
        return "split_not_reconstructible"
    for control_record, control in zip(controls, corpus.controls):
        if control_record.get("control_class") != control.control_class \
                or control_record.get("classifier_term_count") != terms:
            return "split_not_reconstructible"
        tuple_results = control_record.get("tuple_results")
        if not isinstance(tuple_results, list) or len(tuple_results) != len(grid.tuples):
            return "split_not_reconstructible"
        for axes in tuple_results:
            if not isinstance(axes, list) or len(axes) != MAX_AXES_PER_CONTROL:
                return "split_not_reconstructible"
            for axis in axes:
                if axis.get("status") not in ("determined", "abstained"):
                    return "split_not_reconstructible"
    for record, observation in zip(observations, corpus.observations):
        owning_axis = observation.local_axis
        non_owning_axis = "v" if owning_axis == "u" else "u"
        if record.get("local_axis") != owning_axis:
            return "split_not_reconstructible"
        consumed = record.get("consumed_profile_sha256")
        retained = record.get("non_owning_profile_sha256")
        if not _hex64(consumed) or not _hex64(retained) or consumed == retained:
            return "split_not_reconstructible"
        try:
            expected_consumed, _ = _profile_digest(
                observation.patch(), observation_geometry(observation),
                owning_axis)
        except (TypeError, ValueError):
            return "split_not_reconstructible"
        if consumed != expected_consumed:
            return "split_not_reconstructible"
        if retained != observation.non_owning_profile_sha256:
            return "split_not_reconstructible"
        profile = record.get("non_owning_profile")
        if not isinstance(profile, dict):
            return "split_not_reconstructible"
        try:
            profile_payload = dict(profile)
            coverage = dict(profile_payload["coverage"])
            coverage["per_bin"] = tuple(coverage["per_bin"])
            profile_payload["coverage"] = coverage
            rebuilt = _lp.SplitSeamProfileEvidence(**profile_payload)
        except (TypeError, ValueError, KeyError):
            return "split_not_reconstructible"
        if rebuilt.local_axis != non_owning_axis \
                or rebuilt.d_profile_sha256 != retained:
            return "split_not_reconstructible"
    return None


# --- Feasibility and first-feasible selection ------------------------------

def _target_successes(target_power: Fraction, positives: int) -> int:
    """Smallest ``s`` with ``s / positives >= target_power`` (exact ints)."""
    if positives <= 0:
        return 0
    quotient = target_power * positives
    return -((-quotient.numerator) // quotient.denominator)


def tuple_feasible(prediction: dict, grid, tuple_index: int,
                   truth: dict | None = None) -> tuple[bool, dict]:
    """Exact D4 feasibility for one tuple index over a frozen prediction."""
    truth = truth or {}
    successes = 0
    positives = 0
    false_accepts = 0
    for control in prediction["per_control"]:
        result = control["tuple_results"][tuple_index]
        determined = [axis for axis in result if axis["status"] == "determined"]
        if control["control_class"] == "positive":
            positives += 1
            target = truth.get(control["control_id"])
            emitted = [axis.get("emitted_alpha") for axis in result]
            if (target is not None and all(axis["status"] == "determined"
                                           for axis in result)
                    and all(value is not None for value in emitted)
                    and all(tuple(left) == tuple(right)
                            for left, right in zip(emitted, target))):
                successes += 1
        elif determined:
            false_accepts += 1
    target_power = Fraction(int(grid.target_power[0]), int(grid.target_power[1]))
    need = _target_successes(target_power, positives)
    feasible = false_accepts == 0 and successes >= need
    return feasible, {"false_accepts": false_accepts, "successes": successes,
                      "positives": positives, "target_successes": need}


def select_first_feasible(prediction: dict, grid, *,
                          truth: dict | None = None) -> dict:
    """Lexicographically first feasible tuple, or typed ``no_feasible_calibration``."""
    for index in range(len(grid.tuples)):
        feasible, audit = tuple_feasible(prediction, grid, index, truth)
        if feasible:
            return {"status": "selected", "tuple_index": index,
                    "tuple": grid.tuples[index], "audit": audit}
    return {"status": "refused", "reason": "no_feasible_calibration"}


# --- Calibration record construction ---------------------------------------

def build_calibration_record(grid, selected_tuple, *,
                             calibration_corpus: CalibrationCorpus,
                             evaluation_corpus: CalibrationCorpus):
    """Build and validate the accepted ``SplitPhaseCalibration`` record."""
    payload = {
        "calibration_version": SPLIT_PHASE_CALIBRATION_VERSION,
        "partition_config_sha256": _lp.SPLIT_PHASE_PARTITION_CONFIG_SHA256,
        "tau_band": list(selected_tuple[0]),
        "tau_margin": list(selected_tuple[1]),
        "tau_energy": list(selected_tuple[2]),
        "min_held_out_samples": selected_tuple[3],
        "min_distinct_frames_per_axis": selected_tuple[4],
        "tie_break": "no_tie_break_abstain",
        "target_power": list(grid.target_power),
        "calibration_corpus_sha256": calibration_corpus.corpus_sha256,
        "evaluation_corpus_sha256": evaluation_corpus.corpus_sha256,
        "calibration_sample_sha256s": list(calibration_corpus.sample_sha256s),
        "evaluation_sample_sha256s": list(evaluation_corpus.sample_sha256s),
        "null_control_ids": list(calibration_corpus.null_control_ids),
        "seam_control_ids": list(calibration_corpus.seam_control_ids),
        "calibration_grid_sha256": grid.grid_sha256,
    }
    record = _lp.SplitPhaseCalibration(
        calibration_version=payload["calibration_version"],
        partition_config_sha256=payload["partition_config_sha256"],
        tau_band=tuple(payload["tau_band"]),
        tau_margin=tuple(payload["tau_margin"]),
        tau_energy=tuple(payload["tau_energy"]),
        min_held_out_samples=payload["min_held_out_samples"],
        min_distinct_frames_per_axis=payload["min_distinct_frames_per_axis"],
        tie_break=payload["tie_break"],
        target_power=tuple(payload["target_power"]),
        calibration_corpus_sha256=payload["calibration_corpus_sha256"],
        evaluation_corpus_sha256=payload["evaluation_corpus_sha256"],
        calibration_sample_sha256s=tuple(payload["calibration_sample_sha256s"]),
        evaluation_sample_sha256s=tuple(payload["evaluation_sample_sha256s"]),
        null_control_ids=tuple(payload["null_control_ids"]),
        seam_control_ids=tuple(payload["seam_control_ids"]),
        calibration_grid_sha256=payload["calibration_grid_sha256"],
        calibration_sha256=digest_record(payload, "calibration_sha256"))
    refusal = validate_calibration_record_refusal(
        calibration_corpus, evaluation_corpus, grid, record)
    if refusal is not None:
        raise ValueError(refusal)
    return record


def score_freeze(prediction: dict, labels: CalibrationLabels, grid, *,
                 calibration_corpus: CalibrationCorpus,
                 evaluation_corpus: CalibrationCorpus) -> dict:
    """Pure scoring: verify freeze, open-label truth, select, build record."""
    if not verify_prediction_digest(prediction):
        return {"status": "refused", "reason": "split_not_reconstructible"}
    if prediction.get("labels_sha256") != labels.labels_sha256:
        return {"status": "refused", "reason": "split_not_reconstructible"}
    if prediction.get("grid_sha256") != grid.grid_sha256:
        return {"status": "refused", "reason": "split_not_reconstructible"}
    if prediction.get("sigma") != 0:
        return {"status": "refused", "reason": "split_not_reconstructible"}
    if (tuple(labels.null_ids()) != calibration_corpus.null_control_ids
            or tuple(labels.seam_ids()) != calibration_corpus.seam_control_ids):
        return {"status": "refused", "reason": "split_not_reconstructible"}
    if prediction_refusal(prediction, calibration_corpus, grid) is not None:
        return {"status": "refused", "reason": "split_not_reconstructible"}
    outcome = select_first_feasible(prediction, grid, truth=labels.truth())
    if outcome["status"] != "selected":
        return outcome
    try:
        record = build_calibration_record(
            grid, grid.tuples[outcome["tuple_index"]],
            calibration_corpus=calibration_corpus,
            evaluation_corpus=evaluation_corpus)
    except ValueError as error:
        return {"status": "refused", "reason": str(error)}
    return {"status": "selected", "calibration": record,
            "tuple": grid.tuples[outcome["tuple_index"]],
            "tuple_index": outcome["tuple_index"], "audit": outcome["audit"]}


def score_audit_payload(*, sigma: int, split: str, corpus_sha256: str,
                        labels_sha256: str, predictions_sha256: str,
                        frozen_at: str, scored_at: str, outcome: dict,
                        grid_sha256: str, caps) -> dict:
    """Build the ``score_audit_<split>.json`` record (no partial result)."""
    selected = outcome.get("tuple")
    return {
        "audit_version": SPLIT_PHASE_CALIBRATION_AUDIT_VERSION,
        "evaluation_only": True,
        "sigma": sigma,
        "split": split,
        "corpus_sha256": corpus_sha256,
        "labels_sha256": labels_sha256,
        "predictions_sha256": predictions_sha256,
        "grid_sha256": grid_sha256,
        "frozen_at": frozen_at,
        "scored_at": scored_at,
        "scored_after_frozen": bool(scored_at > frozen_at),
        "label_opened_after_prediction_bytes": True,
        "selected_tuple": (None if selected is None else
                           [list(value) if isinstance(value, tuple)
                            else value for value in selected]),
        "outcome": outcome.get("status"),
        "reason": outcome.get("reason"),
        "audit": outcome.get("audit"),
        "caps": caps.to_record(),
        "no_tie_break_abstain": True,
    }


# --- High-level, IO-free synthesis -----------------------------------------

def synthesize_split(sigma: int, *, caps: CalibrationCaps | None = None,
                     ordinals: tuple | None = None) -> dict:
    """Synthesize one split's corpus and sealed labels (no IO).

    The production schedule is the full ``range(96)``; the optional ``ordinals``
    argument exists only so isolated tests can exercise the pure helpers on a
    reduced fixture.  It never changes any frozen formula.
    """
    if caps is None:
        caps = default_caps()
    if sigma not in (0, 1):
        raise ValueError("split_not_reconstructible")
    schedule = tuple(range(96)) if ordinals is None else tuple(ordinals)
    if not schedule or len(schedule) > caps.get("max_controls_per_split"):
        raise ValueError("work_over_cap")
    if len(set(schedule)) != len(schedule) or any(
            type(k) is not int or not 0 <= k < 96 for k in schedule):
        raise ValueError("split_not_reconstructible")
    controls = []
    for k in schedule:
        observations = tuple(
            _make_observation(k, w, j, sigma)
            for w in range(MAX_AXES_PER_CONTROL)
            for j in range(MAX_FRAMES_PER_AXIS))
        controls.append(make_control(k, sigma, observations))
    if len(controls) > caps.get("max_controls_per_split"):
        raise ValueError("work_over_cap")
    if len(controls) * MAX_AXES_PER_CONTROL * MAX_FRAMES_PER_AXIS \
            > caps.get("max_observations_per_split"):
        raise ValueError("work_over_cap")
    if len(controls) * MAX_AXES_PER_CONTROL * MAX_FRAMES_PER_AXIS \
            > caps.get("max_producer_calls_per_split"):
        raise ValueError("work_over_cap")
    patch_bytes = len(controls) * MAX_AXES_PER_CONTROL * MAX_FRAMES_PER_AXIS \
        * SPLIT_PHASE_CALIBRATION_PATCH_SHAPE[0] \
        * SPLIT_PHASE_CALIBRATION_PATCH_SHAPE[1] * 8
    if patch_bytes > caps.get("max_patch_bytes_per_split"):
        raise ValueError("work_over_cap")
    entries = []
    for k in schedule:
        name = class_of_ordinal(k)
        components = (None if name != "positive"
                      else [dyadic(alpha_true(k, w, sigma)) for w in (0, 1, 2)])
        entries.append({"control_id": control_id(sigma, k), "class": name,
                        "components": components,
                        "absence_reason": (None if name == "positive" else name)})
    if len(canonical_json(entries)) > caps.get("max_label_bytes_per_split"):
        raise ValueError("work_over_cap")
    entries = tuple(entries)
    labels = CalibrationLabels(
        entries=entries, labels_sha256=digest_json([dict(item) for item in entries]))
    corpus = make_corpus(tuple(controls), sigma, labels.labels_sha256)
    return {"corpus": corpus, "labels": labels}


def _make_observation(k: int, w: int, j: int, sigma: int) -> CalibrationObservation:
    name = class_of_ordinal(k)
    patch = patch_for(k, w, j, sigma, name)
    axis = local_axis(k, w, j)
    geometry = geometry_for(k, w, sigma)
    owning_digest, owning_bins = _profile_digest(patch, geometry, axis)
    non_owning_axis = "v" if axis == "u" else "u"
    non_owning_digest, _ = _profile_digest(patch, geometry, non_owning_axis)
    if owning_bins != MAX_PROFILE_BINS_PER_OBSERVATION:
        raise ValueError("split_not_reconstructible")
    if owning_digest == non_owning_digest:
        raise ValueError("split_not_reconstructible")
    is_seam = name == "positive"
    components = (tuple(dyadic(alpha_true(k, other, sigma))
                        for other in (0, 1, 2)) if is_seam else None)
    payload = {
        "observation_id": observation_id(sigma, k, w, j),
        "sigma": sigma,
        "split": SPLIT_PHASE_CALIBRATION_SPLITS[sigma],
        "control_id": control_id(sigma, k),
        "control_class": name,
        "ordinal": k,
        "patch_sha256": patch_sha256(patch),
        "L_dyadic": tuple(dyadic(leading_cell(k, w, j, sigma))),
        "T_dyadic": tuple(dyadic(frame_translation(k, w, j, sigma, name))),
        "alpha_dyadic_hidden": tuple(dyadic(alpha_true(k, w, sigma))),
        "H_w": tuple(tuple(dyadic(value)) for value in hypothesis_set(k, w, sigma)),
        "selected_b": selected_b(k, w, sigma),
        "selected_b_ratio": tuple(int(part) for part in
                                  selected_b(k, w, sigma).as_integer_ratio()),
        "period_pixels": period_pixels(k, w, sigma),
        "local_axis": axis,
        "world_axis": w,
        "world_sign": world_sign(k, w),
        "boundary_contrast": boundary_contrast(k, w),
        "nuisance_amplitude": nuisance_amplitude(k, w),
        "geometry": geometry.to_record(),
        "geometry_sha256": digest_json(geometry.to_record()),
        "patch_float64_le_base64": base64.b64encode(
            patch.tobytes(order="C")).decode("ascii"),
        "non_owning_profile_sha256": non_owning_digest,
    }
    payload["provenance_sha256"] = digest_record(
        _provenance_preimage(payload), "provenance_sha256")
    payload["label_payload_sha256"] = label_payload_sha256(
        payload["control_id"], name, components)
    payload["observation_sha256"] = digest_record(payload, "observation_sha256")
    return CalibrationObservation(**payload)


def _provenance_preimage(payload: dict) -> dict:
    return {
        "sigma": payload["sigma"], "split": payload["split"],
        "control_id": payload["control_id"],
        "observation_id": payload["observation_id"],
        "L_dyadic": list(payload["L_dyadic"]),
        "T_dyadic": list(payload["T_dyadic"]),
        "H_w": [list(value) for value in payload["H_w"]],
        "selected_b_ratio": list(payload["selected_b_ratio"]),
        "period_pixels": payload["period_pixels"],
        "a": payload["local_axis"], "w": payload["world_axis"],
        "s": payload["world_sign"],
        "geometry_sha256": payload["geometry_sha256"],
        "boundary_contrast": payload["boundary_contrast"],
        "nuisance_amplitude": payload["nuisance_amplitude"],
    }
