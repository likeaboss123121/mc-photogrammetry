"""Isolated V2-B physical phase-anchor determination (Path B) and diagnostics.

This module implements the frozen ``M4-V2B-physical-phase-anchor-contract``
(F7C336E8...426787E) on top of the accepted P0 split-phase producer
(``lattice_planes.produce_split_phase_evidence``), the accepted V2-A endpoint
provenance result, and the accepted C2-R-B relative-translation certificate.

Path B is the only path that may determine ``alpha``: the global block-grid
phase/seam anchor in the C2-R-B canonical gauge.  It uses only the held-out
``D`` profile produced by P0 and the accepted calibration.  Path A (retained
full-patch P1-A evidence) is a consistency-only, non-promotable diagnostic: it
never emits ``determined`` and never sets ``alpha``.

The module performs no IO and has no filesystem, network, process, environment,
or dynamic-import access.  The accepted empirical calibration is embedded as a
versioned built-in record; no label, corpus, or grid file is read at run time.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, replace as _dataclass_replace
from fractions import Fraction
import hashlib
import json
import math
import numpy as np
from typing import Any, Mapping, Sequence

from mc_photogrammetry.reconstruction import lattice_planes
from mc_photogrammetry.reconstruction import plane_axis_support
from mc_photogrammetry.reconstruction import plane_endpoint_provenance as v2a
from mc_photogrammetry.reconstruction import plane_graph
from mc_photogrammetry.reconstruction import plane_relative_translation
from mc_photogrammetry.reconstruction import plane_residue
from mc_photogrammetry.reconstruction import plane_shift_domain
from mc_photogrammetry.reconstruction import plane_vector_domain


V2B_ANCHOR_VERSION = "v2b-physical-phase-anchor-1"
V2B_ANCHOR_CALIBRATION_VERSION = "v2b-anchor-calibration-1"
V2B_ANCHOR_REFUSAL_DETAIL_MAX_BYTES = 4096
V2B_ANCHOR_SCHEMA = "v2b-anchor-result-1"

V2B_ANCHOR_REFUSAL_REASONS: tuple[str, ...] = (
    "v2b_input_invalid",
    "c2rb_replay_mismatch",
    "v2a_replay_mismatch",
    "p0_patch_unavailable",
    "p0_patch_mismatch",
    "split_not_reconstructible",
    "split_replay_mismatch",
    "p0_leakage",
    "g_phase_reconstruction_refused",
    "p1b_source_mismatch",
    "seam_profile_not_reconstructible",
    "seam_profile_payload_invalid",
    "split_residue_binding_mismatch",
    "translation_class_not_unique",
    "certificate_mismatch",
    "propagation_unavailable",
    "propagation_origin_inconsistent",
    "axis_transform_invalid",
    "hypothesis_set_too_large",
    "noncanonical_dyadic",
    "refusal_detail_too_large",
    "budget_exhausted",
    "work_over_cap",
    "calibration_not_builtin",
    "legacy_full_patch_nonpromotable",
    "result_validation_mismatch",
)
"""Closed V2-B refusal enum; no other string is ever a refusal reason."""

V2B_ABSTENTION_REASONS: tuple[str, ...] = (
    "endpoint_multiple",
    "endpoint_none",
    "input_gate_not_unique",
    "no_hypothesis",
    "no_seam_evidence",
    "zero_coverage",
    "multiple_hypotheses_supported",
    "evidence_conflict",
    "propagation_unavailable",
    "axis_transform_invalid",
    "insufficient_data",
)
"""Typed per-axis / overall abstention reasons."""

V2B_PREDICTION_REASONS: tuple[str, ...] = (
    V2B_ABSTENTION_REASONS + ("determined", "partial"))
"""The full closed prediction-reason tuple (determined/partial plus abstentions)."""

# Accepted empirical split-phase calibration, embedded as literals.  The
# accepted record SHA-256 commits to every field that is not repeated here
# (corpus sample hashes, control ids), so binding the digest plus the compact
# numeric fields binds the whole accepted record without any run-time IO.
_SOURCE_CALIBRATION_SHA256 = \
    "dc4a3ac698c80c915cd4e61933044c37766e5be394e56d220e5bda80d2fbbc06"
_BOUND_PARTITION_SHA256 = \
    "b2225d8f2ca16bf01189826870511e025e80c108ac3c1122c57ed4f7ba6b214f"
_CALIBRATION_GRID_SHA256 = \
    "87c8d3e1e00a91b303a42633c45b7684939b47d8dccd10294ddb96c03a3b39fe"
_CALIBRATION_CORPUS_SHA256 = \
    "45b1b21a1e59510d4dfa382054665d3700b55decfb0b69660f80de9572baa72b"
_EVALUATION_CORPUS_SHA256 = \
    "062c41abab5c4bc6f124cf2f6f349f40a87c6aefe93623d724234d6ceac06528"
_NULL_CONTROL_IDS_SHA256 = \
    "d26dc18ba62457dfb0de50b4894c3c82f09e5cfbded81841f3a6cd4d205af04d"
_SEAM_CONTROL_IDS_SHA256 = \
    "bfd7d3d12a37c195fe5480ca4718ad7b57cafa1b4eef4297244d90f2d4d7160e"
_CALIBRATION_SAMPLE_ROOT_SHA256 = \
    "37d17bf0b5948a5fa9b0e4a1e33d94fefabbf15e7a95ec6fac159d8b32dde42f"
_EVALUATION_SAMPLE_ROOT_SHA256 = \
    "8a05721992f705fb8edea7cd95018e63a7852ca5a656cff2505520b5a23e69e4"
_TAU_BAND = (1, 8)
_TAU_MARGIN = (1, 8)
_TAU_ENERGY = (0, 1)
_MIN_HELD_OUT_SAMPLES = 1
_MIN_DISTINCT_FRAMES_PER_AXIS = 2
_TIE_BREAK = "no_tie_break_abstain"
_TARGET_POWER = (1, 2)

_PHASE_UNAVAILABLE_REASONS = frozenset({
    "insufficient_period", "empty_profile", "nonfinite_profile",
    "zero_resultant", "invalid_profile", "invalid_period",
})
_WORLD_AXES = (0, 1, 2)
_LOCAL_AXES = ("u", "v")


# ---------------------------------------------------------------------------
# Canonical JSON, digest, exact-rational helpers
# ---------------------------------------------------------------------------

def V2B_CANONICAL_JSON(value: Any) -> bytes:
    """Frozen canonical JSON encoding used for every V2-B digest."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(V2B_CANONICAL_JSON(value)).hexdigest()


def _hex64(value: Any) -> bool:
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _strict_int(value: Any, *, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _finite_float(value: Any) -> float | None:
    if type(value) not in (int, float) or type(value) is bool:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _fraction_from_float(value: float) -> Fraction:
    return Fraction(*float(value).as_integer_ratio())


def _reduce_dyadic(value: Fraction) -> tuple[int, int]:
    """Return the canonical reduced dyadic of ``value mod 1``.

    Every finite IEEE-754 float is a dyadic rational, so every accepted
    translation, leading edge, origin and anchor is exactly dyadic.  The only
    accepted representations are the canonical zero ``(0, 1)`` and an odd
    numerator strictly between zero and a power-of-two denominator.
    """
    reduced = Fraction(value) % 1
    if reduced == 0:
        return (0, 1)
    numerator, denominator = reduced.numerator, reduced.denominator
    if numerator < 0 or numerator >= denominator:
        raise ValueError("noncanonical_dyadic")
    if denominator & (denominator - 1):
        raise ValueError("noncanonical_dyadic")
    if numerator % 2 == 0:
        raise ValueError("noncanonical_dyadic")
    return (numerator, denominator)


def _validate_dyadic(value: Any) -> bool:
    try:
        return _reduce_dyadic(_fraction(value)) == tuple(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return False


def _fraction(value: tuple[int, int]) -> Fraction:
    if (not isinstance(value, tuple) or len(value) != 2
            or any(type(part) is not int for part in value)):
        raise ValueError("noncanonical_dyadic")
    numerator, denominator = value
    if numerator == 0:
        if denominator != 1:
            raise ValueError("noncanonical_dyadic")
        return Fraction(0)
    if (denominator <= 0 or denominator & (denominator - 1)
            or numerator < 0 or numerator >= denominator or numerator % 2 == 0):
        raise ValueError("noncanonical_dyadic")
    return Fraction(numerator, denominator)


def _reduce_fraction(value: Fraction) -> tuple[int, int]:
    if value == 0:
        return (0, 1)
    return (value.numerator, value.denominator)


def _validate_reduced(value: Any) -> bool:
    if (not isinstance(value, tuple) or len(value) != 2
            or any(type(part) is not int for part in value)):
        return False
    if value[0] == 0:
        return value[1] == 1
    return value[1] > 0 and math.gcd(abs(value[0]), value[1]) == 1


def _frac(value: tuple[int, int]) -> Fraction:
    if not _validate_reduced(value):
        raise ValueError("noncanonical_dyadic")
    return Fraction(value[0], value[1])


def _signed_dyadic(value: Any) -> Fraction:
    """Parse one entry of the C2-R-B full signed dyadic translation domain.

    C2-R-B emits exact reduced dyadics with a positive power-of-two denominator
    and without any modulo-one normalization, so integral values such as ``3``
    and negative fractions such as ``-1/4`` are valid translations.  This parser
    accepts exactly that domain -- it is the shared representation contract of
    ``plane_relative_translation._validate_dyadic`` -- and still refuses every
    non-reduced, non-dyadic or malformed tuple as ``noncanonical_dyadic``.
    Periodicity is applied later, at the exact-rational anchor, never here.
    """
    if (not isinstance(value, tuple) or len(value) != 2
            or any(type(part) is not int for part in value)):
        raise ValueError("noncanonical_dyadic")
    numerator, denominator = value
    if (denominator <= 0 or denominator & (denominator - 1)
            or math.gcd(abs(numerator), denominator) != 1):
        raise ValueError("noncanonical_dyadic")
    return Fraction(numerator, denominator)


def _mod1(value: Fraction) -> Fraction:
    return value - math.floor(value)


def _d1(first: Fraction, second: Fraction) -> Fraction:
    difference = abs(first - second) % 1
    return min(difference, 1 - difference)


def _dtorus(first: Sequence[Fraction], second: Sequence[Fraction]) -> Fraction:
    return max(_d1(left, right) for left, right in zip(first, second))


def _detail(value: Any) -> Any:
    """Return the canonical JSON-safe refusal detail, or raise."""
    if value is None or type(value) is bool or type(value) is int or type(value) is str:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("v2b_input_invalid")
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_detail(item) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("v2b_input_invalid")
            result[key] = _detail(item)
        return result
    raise ValueError("v2b_input_invalid")


def _seal(cls: type, fields: Mapping[str, Any], digest_field: str) -> Any:
    provisional = object.__new__(cls)
    for name, value in fields.items():
        object.__setattr__(provisional, name, value)
    object.__setattr__(provisional, digest_field, "")
    payload = provisional.to_record()
    payload.pop(digest_field)
    object.__setattr__(provisional, digest_field, _sha256(payload))
    cls.__post_init__(provisional)
    return provisional


# ---------------------------------------------------------------------------
# Budgets and work
# ---------------------------------------------------------------------------

_BUDGET_FIELDS: tuple[str, ...] = (
    "max_readings", "max_observations", "max_frames",
    "max_hypotheses_per_axis", "max_hypotheses_total",
    "max_contributor_refs_per_hypothesis", "max_contributor_refs_total",
    "max_source_records", "max_boundary_interior_terms", "max_d_profile_terms",
    "max_payload_bytes", "max_replay_crosslinks", "max_pixel_samples",
    "max_held_out_samples_per_reading", "max_split_reconstructions",
    "max_accepted_frame_supports",
)
_WORK_FIELDS: tuple[str, ...] = (
    "readings", "observations", "frames", "hypotheses_per_axis",
    "hypotheses_total", "contributor_refs_per_hypothesis",
    "contributor_refs_total", "source_records", "boundary_interior_terms",
    "d_profile_terms", "payload_bytes", "replay_crosslinks", "pixel_samples",
    "held_out_samples_per_reading", "split_reconstructions",
    "accepted_frame_supports",
)
_WORK_TO_CAP = dict(zip(_WORK_FIELDS, _BUDGET_FIELDS))
_CAP_TO_WORK = {cap: work for work, cap in _WORK_TO_CAP.items()}
# Amended V2-B contract (§10): preflight enforces the fifteen existing static
# counters only.  ``accepted_frame_supports`` carries the truthful planned upper
# bound ``U`` and is never compared against ``max_accepted_frame_supports`` at
# preflight; the runtime counter ``K`` is checked at support insertion instead.
_STATIC_WORK_FIELDS: tuple[str, ...] = tuple(
    name for name in _WORK_FIELDS if name != "accepted_frame_supports")
# Caps a deterministic planned preflight can legitimately exhaust.  The amended
# ``max_accepted_frame_supports`` allowance is runtime-only: its work counter is
# excluded from ``_STATIC_WORK_FIELDS`` and ``_plan_work`` never compares it, so
# a planned branch that names it is a forgery and is refused by the result
# constructor.
_PLANNED_EXHAUSTIBLE_CAPS: frozenset[str] = frozenset(
    _WORK_TO_CAP[name] for name in _STATIC_WORK_FIELDS)
# Counters that a completed prediction must report identically in its planned
# and runtime snapshots because the runtime pass processes the whole plan.
_PLANNED_RUNTIME_SHARED_FIELDS: tuple[str, ...] = (
    "readings", "frames", "hypotheses_per_axis", "hypotheses_total",
    "contributor_refs_per_hypothesis", "contributor_refs_total",
    "source_records", "d_profile_terms", "payload_bytes",
    "replay_crosslinks", "pixel_samples", "split_reconstructions",
)
# A completed run additionally processes every planned observation key, so the
# per-reading held-out maximum is identical in both snapshots.
_PREDICTION_SHARED_FIELDS: tuple[str, ...] = \
    _PLANNED_RUNTIME_SHARED_FIELDS + ("held_out_samples_per_reading",)
# The only keys a budget-exhaustion refusal detail may carry.  Anything else
# would smuggle hypothesis, observation, or anchor evidence into a branch that
# the contract requires to be evidence-free.
_BUDGET_DETAIL_KEYS = frozenset((
    "budget_stage", "cap", "limit", "observed", "planned_work",
    "runtime_work",
))


@dataclass(frozen=True, slots=True)
class V2BAnchorBudget:
    """Explicit positive exact-integer caps, no defaults."""

    max_readings: int
    max_observations: int
    max_frames: int
    max_hypotheses_per_axis: int
    max_hypotheses_total: int
    max_contributor_refs_per_hypothesis: int
    max_contributor_refs_total: int
    max_source_records: int
    max_boundary_interior_terms: int
    max_d_profile_terms: int
    max_payload_bytes: int
    max_replay_crosslinks: int
    max_pixel_samples: int
    max_held_out_samples_per_reading: int
    max_split_reconstructions: int
    max_accepted_frame_supports: int
    budget_sha256: str

    def __post_init__(self) -> None:
        for name in _BUDGET_FIELDS:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError("v2b_input_invalid")
        if (not _hex64(self.budget_sha256)
                or self.budget_sha256 != _sha256(self._payload())):
            raise ValueError("v2b_input_invalid")

    def _payload(self) -> dict:
        return {name: getattr(self, name) for name in _BUDGET_FIELDS}

    def to_record(self) -> dict:
        record = self._payload()
        record["budget_sha256"] = self.budget_sha256
        return record


def make_anchor_budget(**caps: int) -> V2BAnchorBudget:
    """Build a self-validating budget from explicit caps (no defaults).

    Every cap must already be an exact positive ``int``.  Booleans, strings,
    floats, ``None`` and every other non-exact-integer value are rejected
    *before* any coercion, so a public caller can never smuggle ``True``,
    ``"9"`` or ``9.0`` past the constructor and have it silently become ``9``.
    """
    payload: dict[str, int] = {}
    for name in _BUDGET_FIELDS:
        if name not in caps:
            raise ValueError("v2b_input_invalid")
        value = caps[name]
        if type(value) is not int or value < 1:
            raise ValueError("v2b_input_invalid")
        payload[name] = value
    return V2BAnchorBudget(**payload,
                           budget_sha256=_sha256(payload))


@dataclass(frozen=True, slots=True)
class V2BAnchorWork:
    """Exact nonnegative counters, one-to-one with the budget caps."""

    readings: int
    observations: int
    frames: int
    hypotheses_per_axis: int
    hypotheses_total: int
    contributor_refs_per_hypothesis: int
    contributor_refs_total: int
    source_records: int
    boundary_interior_terms: int
    d_profile_terms: int
    payload_bytes: int
    replay_crosslinks: int
    pixel_samples: int
    held_out_samples_per_reading: int
    split_reconstructions: int
    accepted_frame_supports: int
    work_sha256: str

    def __post_init__(self) -> None:
        for name in _WORK_FIELDS:
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError("v2b_input_invalid")
        # Exact alias relations that hold on every branch of the accepted
        # implementation: each processed observation contributes one boundary
        # and one interior term, and each accepted source record contributes
        # exactly two replay cross-links.  A freshly resealed record whose
        # counters disagree is refused before any stale digest could hide it.
        if self.boundary_interior_terms != 2 * self.observations:
            raise ValueError("v2b_input_invalid")
        if self.replay_crosslinks != 2 * self.source_records:
            raise ValueError("v2b_input_invalid")
        if (not _hex64(self.work_sha256)
                or self.work_sha256 != _sha256(self._payload())):
            raise ValueError("v2b_input_invalid")

    def _payload(self) -> dict:
        return {name: getattr(self, name) for name in _WORK_FIELDS}

    def to_record(self) -> dict:
        record = self._payload()
        record["work_sha256"] = self.work_sha256
        return record


def _make_work(counters: Mapping[str, int]) -> V2BAnchorWork:
    payload: dict[str, int] = {}
    for name in _WORK_FIELDS:
        value = counters.get(name, 0)
        if type(value) is not int:
            raise ValueError("v2b_input_invalid")
        payload[name] = value
    return V2BAnchorWork(**payload, work_sha256=_sha256(payload))


def _work_within_cap(work: V2BAnchorWork, budget: V2BAnchorBudget) -> bool:
    """True when every preflight-enforced static counter is within its cap.

    ``accepted_frame_supports`` is deliberately excluded: the amended contract
    records the planned upper bound ``U`` without comparing it to the runtime
    allowance.
    """
    return all(getattr(work, name) <= getattr(budget, _WORK_TO_CAP[name])
               for name in _STATIC_WORK_FIELDS)


def _first_over_cap(work: V2BAnchorWork, budget: V2BAnchorBudget) -> str | None:
    """First preflight-enforced static counter strictly above its cap."""
    for name in _STATIC_WORK_FIELDS:
        if getattr(work, name) > getattr(budget, _WORK_TO_CAP[name]):
            return _WORK_TO_CAP[name]
    return None


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class V2BAnchorRefusal:
    """Typed, deterministically ordered, bounded refusal."""

    reason: str
    detail: Any
    detail_sha256: str | None
    refusal_sha256: str

    def __post_init__(self) -> None:
        if type(self.reason) is not str or self.reason not in V2B_ANCHOR_REFUSAL_REASONS:
            raise ValueError("v2b_input_invalid")
        normalized = _detail(self.detail)
        encoded = V2B_CANONICAL_JSON(normalized)
        if len(encoded) > V2B_ANCHOR_REFUSAL_DETAIL_MAX_BYTES:
            raise ValueError("refusal_detail_too_large")
        expected = None if normalized is None else hashlib.sha256(encoded).hexdigest()
        if self.detail_sha256 != expected:
            raise ValueError("v2b_input_invalid")
        payload = {"reason": self.reason, "detail_sha256": self.detail_sha256}
        if (not _hex64(self.refusal_sha256)
                or self.refusal_sha256 != _sha256(payload)):
            raise ValueError("v2b_input_invalid")
        object.__setattr__(self, "detail", normalized)

    def to_record(self) -> dict:
        return {
            "reason": self.reason,
            "detail": self.detail,
            "detail_sha256": self.detail_sha256,
            "refusal_sha256": self.refusal_sha256,
        }


def _make_refusal(reason: str, detail: Any = None, cap: str | None = None) -> V2BAnchorRefusal:
    if reason not in V2B_ANCHOR_REFUSAL_REASONS:
        raise ValueError("v2b_input_invalid")
    normalized = _detail(detail)
    encoded = V2B_CANONICAL_JSON(normalized)
    if len(encoded) > V2B_ANCHOR_REFUSAL_DETAIL_MAX_BYTES:
        normalized = {"limit": V2B_ANCHOR_REFUSAL_DETAIL_MAX_BYTES,
                      "observed": len(encoded), "cap": cap}
        reason = "refusal_detail_too_large"
        encoded = V2B_CANONICAL_JSON(normalized)
    detail_sha = None if normalized is None else hashlib.sha256(encoded).hexdigest()
    return V2BAnchorRefusal(reason, normalized, detail_sha,
                            _sha256({"reason": reason, "detail_sha256": detail_sha}))


# ---------------------------------------------------------------------------
# Calibration binding
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class V2BAnchorCalibration:
    """Bound copy of the accepted P0 SplitPhaseCalibration plus replay context."""

    calibration_version: str
    source_calibration_sha256: str
    partition_config_sha256: str
    tau_band: tuple[int, int]
    tau_margin: tuple[int, int]
    tau_energy: tuple[int, int]
    min_held_out_samples: int
    min_distinct_frames_per_axis: int
    tie_break: str
    target_power: tuple[int, int]
    calibration_grid_sha256: str
    calibration_corpus_sha256: str
    evaluation_corpus_sha256: str
    null_control_ids_sha256: str
    seam_control_ids_sha256: str
    calibration_sample_root_sha256: str
    evaluation_sample_root_sha256: str
    replay_comparison_tolerance: float
    replay_closure_tolerance: float
    calibration_sha256: str

    def __post_init__(self) -> None:
        if (self.calibration_version != V2B_ANCHOR_CALIBRATION_VERSION
                or self.tie_break != _TIE_BREAK):
            raise ValueError("calibration_not_builtin")
        for name in ("tau_band", "tau_margin", "tau_energy", "target_power"):
            object.__setattr__(self, name, _canonical_dyadic(getattr(self, name)))
        if self.tau_margin == (0, 1) or self.tau_band[0] != _TAU_BAND[0] \
                or self.tau_band[1] != _TAU_BAND[1]:
            raise ValueError("calibration_not_builtin")
        if (type(self.min_held_out_samples) is not int
                or self.min_held_out_samples < 1
                or type(self.min_distinct_frames_per_axis) is not int
                or self.min_distinct_frames_per_axis < 2):
            raise ValueError("calibration_not_builtin")
        for name in ("source_calibration_sha256", "partition_config_sha256",
                     "calibration_grid_sha256", "calibration_corpus_sha256",
                     "evaluation_corpus_sha256", "null_control_ids_sha256",
                     "seam_control_ids_sha256", "calibration_sample_root_sha256",
                     "evaluation_sample_root_sha256"):
            if not _hex64(getattr(self, name)):
                raise ValueError("calibration_not_builtin")
        for name in ("replay_comparison_tolerance", "replay_closure_tolerance"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(float(value)) \
                    or float(value) < 0.0:
                raise ValueError("calibration_not_builtin")
            object.__setattr__(self, name, float(value))
        if (not _hex64(self.calibration_sha256)
                or self.calibration_sha256 != _sha256(self._payload())):
            raise ValueError("calibration_not_builtin")
        if (self.source_calibration_sha256 != _SOURCE_CALIBRATION_SHA256
                or self.partition_config_sha256 != _BOUND_PARTITION_SHA256):
            raise ValueError("calibration_not_builtin")

    def _payload(self) -> dict:
        return {
            "calibration_version": self.calibration_version,
            "source_calibration_sha256": self.source_calibration_sha256,
            "partition_config_sha256": self.partition_config_sha256,
            "tau_band": list(self.tau_band),
            "tau_margin": list(self.tau_margin),
            "tau_energy": list(self.tau_energy),
            "min_held_out_samples": self.min_held_out_samples,
            "min_distinct_frames_per_axis": self.min_distinct_frames_per_axis,
            "tie_break": self.tie_break,
            "target_power": list(self.target_power),
            "calibration_grid_sha256": self.calibration_grid_sha256,
            "calibration_corpus_sha256": self.calibration_corpus_sha256,
            "evaluation_corpus_sha256": self.evaluation_corpus_sha256,
            "null_control_ids_sha256": self.null_control_ids_sha256,
            "seam_control_ids_sha256": self.seam_control_ids_sha256,
            "calibration_sample_root_sha256": self.calibration_sample_root_sha256,
            "evaluation_sample_root_sha256": self.evaluation_sample_root_sha256,
            "replay_comparison_tolerance": self.replay_comparison_tolerance,
            "replay_closure_tolerance": self.replay_closure_tolerance,
        }

    def to_record(self) -> dict:
        record = self._payload()
        record["calibration_sha256"] = self.calibration_sha256
        return record

    @property
    def tau_band_fraction(self) -> Fraction:
        return _frac(self.tau_band)

    @property
    def tau_margin_fraction(self) -> Fraction:
        return _frac(self.tau_margin)

    @property
    def tau_energy_fraction(self) -> Fraction:
        return _frac(self.tau_energy)

    @classmethod
    def from_split_phase_calibration(
        cls, record: "lattice_planes.SplitPhaseCalibration",
    ) -> "V2BAnchorCalibration":
        """Bind the accepted P0 calibration without any run-time label access."""
        if not isinstance(record, lattice_planes.SplitPhaseCalibration):
            raise ValueError("calibration_not_builtin")
        if record.calibration_sha256 != _SOURCE_CALIBRATION_SHA256:
            raise ValueError("calibration_not_builtin")
        if (record.partition_config_sha256 != _BOUND_PARTITION_SHA256
                or record.calibration_grid_sha256 != _CALIBRATION_GRID_SHA256
                or record.calibration_corpus_sha256 != _CALIBRATION_CORPUS_SHA256
                or record.evaluation_corpus_sha256 != _EVALUATION_CORPUS_SHA256
                or tuple(record.tau_band) != _TAU_BAND
                or tuple(record.tau_margin) != _TAU_MARGIN
                or tuple(record.tau_energy) != _TAU_ENERGY
                or record.min_held_out_samples != _MIN_HELD_OUT_SAMPLES
                or record.min_distinct_frames_per_axis != _MIN_DISTINCT_FRAMES_PER_AXIS
                or record.tie_break != _TIE_BREAK
                or tuple(record.target_power) != _TARGET_POWER):
            raise ValueError("calibration_not_builtin")
        if (_sha256(list(record.null_control_ids)) != _NULL_CONTROL_IDS_SHA256
                or _sha256(list(record.seam_control_ids)) != _SEAM_CONTROL_IDS_SHA256):
            raise ValueError("calibration_not_builtin")
        if (_sha256(list(record.calibration_sample_sha256s))
                != _CALIBRATION_SAMPLE_ROOT_SHA256
                or _sha256(list(record.evaluation_sample_sha256s))
                != _EVALUATION_SAMPLE_ROOT_SHA256):
            raise ValueError("calibration_not_builtin")
        return V2B_BUILTIN_CALIBRATION


def _canonical_dyadic(value: Any) -> tuple[int, int]:
    """Validate a reduced-dyadic pair, canonicalizing float-free ints."""
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or any(type(part) is not int for part in value)):
        raise ValueError("noncanonical_dyadic")
    value = (value[0], value[1])
    if value == (0, 1):
        return (0, 1)
    if _validate_dyadic(value):
        return (value[0], value[1])
    if value == (1, 1):
        return (1, 1)
    raise ValueError("noncanonical_dyadic")


V2B_BUILTIN_CALIBRATION = None  # assigned below
_V2B_CALIBRATION_PAYLOAD = {
    "calibration_version": V2B_ANCHOR_CALIBRATION_VERSION,
    "source_calibration_sha256": _SOURCE_CALIBRATION_SHA256,
    "partition_config_sha256": _BOUND_PARTITION_SHA256,
    "tau_band": list(_TAU_BAND),
    "tau_margin": list(_TAU_MARGIN),
    "tau_energy": list(_TAU_ENERGY),
    "min_held_out_samples": _MIN_HELD_OUT_SAMPLES,
    "min_distinct_frames_per_axis": _MIN_DISTINCT_FRAMES_PER_AXIS,
    "tie_break": _TIE_BREAK,
    "target_power": list(_TARGET_POWER),
    "calibration_grid_sha256": _CALIBRATION_GRID_SHA256,
    "calibration_corpus_sha256": _CALIBRATION_CORPUS_SHA256,
    "evaluation_corpus_sha256": _EVALUATION_CORPUS_SHA256,
    "null_control_ids_sha256": _NULL_CONTROL_IDS_SHA256,
    "seam_control_ids_sha256": _SEAM_CONTROL_IDS_SHA256,
    "calibration_sample_root_sha256": _CALIBRATION_SAMPLE_ROOT_SHA256,
    "evaluation_sample_root_sha256": _EVALUATION_SAMPLE_ROOT_SHA256,
    "replay_comparison_tolerance": 1e-12,
    "replay_closure_tolerance": 0.0,
}
V2B_BUILTIN_CALIBRATION = V2BAnchorCalibration(
    **_V2B_CALIBRATION_PAYLOAD,
    calibration_sha256=_sha256(_V2B_CALIBRATION_PAYLOAD),
)


# ---------------------------------------------------------------------------
# Provenance contributors and hypotheses
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class V2BAnchorContributor:
    """One ordered contributor to one candidate seam in ``H_w``."""

    source_kind: str
    source_record_sha256: str
    domain_sha256: str | None
    local_axis: str | None
    local_axis_label: int | None
    absolute_quadrant: int | None
    world_sign: int
    frame_translation_dyadic: tuple[int, int]
    leading_cell_coordinate: tuple[int, int] | None
    selected_b: tuple[int, int]
    residue_index: int | None
    source_hypothesis_sha256: str | None
    contributor_order: int
    contributor_sha256: str

    def __post_init__(self) -> None:
        if self.source_kind not in ("reading_domain", "phase_candidates",
                                    "edge_reference"):
            raise ValueError("v2b_input_invalid")
        if not _hex64(self.source_record_sha256):
            raise ValueError("v2b_input_invalid")
        if self.domain_sha256 is not None and not _hex64(self.domain_sha256):
            raise ValueError("v2b_input_invalid")
        if self.local_axis is not None and self.local_axis not in _LOCAL_AXES:
            raise ValueError("v2b_input_invalid")
        if self.local_axis_label is not None and (
                type(self.local_axis_label) is not int
                or self.local_axis_label not in (0, 1, 2)):
            raise ValueError("v2b_input_invalid")
        if self.absolute_quadrant is not None and (
                type(self.absolute_quadrant) is not int
                or self.absolute_quadrant not in (0, 1, 2, 3)):
            raise ValueError("v2b_input_invalid")
        if self.world_sign not in (-1, 1):
            raise ValueError("axis_transform_invalid")
        if not _validate_dyadic(self.frame_translation_dyadic):
            raise ValueError("noncanonical_dyadic")
        if (self.leading_cell_coordinate is not None
                and not _validate_dyadic(self.leading_cell_coordinate)):
            raise ValueError("noncanonical_dyadic")
        if not _validate_reduced(self.selected_b):
            raise ValueError("noncanonical_dyadic")
        if self.residue_index is not None and not _strict_int(self.residue_index):
            raise ValueError("v2b_input_invalid")
        if (self.source_hypothesis_sha256 is not None
                and not _hex64(self.source_hypothesis_sha256)):
            raise ValueError("v2b_input_invalid")
        if not _strict_int(self.contributor_order):
            raise ValueError("v2b_input_invalid")
        if (not _hex64(self.contributor_sha256)
                or self.contributor_sha256 != _sha256(self._payload())):
            raise ValueError("v2b_input_invalid")

    def _payload(self) -> dict:
        return {
            "source_kind": self.source_kind,
            "source_record_sha256": self.source_record_sha256,
            "domain_sha256": self.domain_sha256,
            "local_axis": self.local_axis,
            "local_axis_label": self.local_axis_label,
            "absolute_quadrant": self.absolute_quadrant,
            "world_sign": self.world_sign,
            "frame_translation_dyadic": list(self.frame_translation_dyadic),
            "leading_cell_coordinate": (None if self.leading_cell_coordinate is None
                                        else list(self.leading_cell_coordinate)),
            "selected_b": list(self.selected_b),
            "residue_index": self.residue_index,
            "source_hypothesis_sha256": self.source_hypothesis_sha256,
            "contributor_order": self.contributor_order,
        }

    def to_record(self) -> dict:
        record = self._payload()
        record["contributor_sha256"] = self.contributor_sha256
        return record


def _make_contributor(**fields: Any) -> V2BAnchorContributor:
    payload = {
        "source_kind": fields["source_kind"],
        "source_record_sha256": fields["source_record_sha256"],
        "domain_sha256": fields.get("domain_sha256"),
        "local_axis": fields.get("local_axis"),
        "local_axis_label": fields.get("local_axis_label"),
        "absolute_quadrant": fields.get("absolute_quadrant"),
        "world_sign": fields["world_sign"],
        "frame_translation_dyadic": fields["frame_translation_dyadic"],
        "leading_cell_coordinate": fields.get("leading_cell_coordinate"),
        "selected_b": fields["selected_b"],
        "residue_index": fields.get("residue_index"),
        "source_hypothesis_sha256": fields.get("source_hypothesis_sha256"),
        "contributor_order": fields["contributor_order"],
    }
    return V2BAnchorContributor(
        **payload, contributor_sha256=_sha256(payload))


@dataclass(frozen=True, slots=True)
class V2BAnchorHypothesis:
    """One distinct candidate seam with its complete ordered contributors."""

    world_axis: int
    numerator: int
    denominator: int
    contributors: tuple[V2BAnchorContributor, ...]
    contributor_count: int
    hypothesis_sha256: str

    def __post_init__(self) -> None:
        if type(self.world_axis) is not int or self.world_axis not in _WORLD_AXES:
            raise ValueError("v2b_input_invalid")
        if _reduce_dyadic(Fraction(self.numerator, self.denominator)) \
                != (self.numerator, self.denominator):
            raise ValueError("noncanonical_dyadic")
        if (not isinstance(self.contributors, tuple) or not self.contributors
                or any(not isinstance(value, V2BAnchorContributor)
                       for value in self.contributors)
                or type(self.contributor_count) is not int
                or self.contributor_count != len(self.contributors)):
            raise ValueError("v2b_input_invalid")
        if tuple(value.contributor_order for value in self.contributors) \
                != tuple(range(len(self.contributors))):
            raise ValueError("v2b_input_invalid")
        if (not _hex64(self.hypothesis_sha256)
                or self.hypothesis_sha256 != _sha256(self._payload())):
            raise ValueError("v2b_input_invalid")

    @property
    def value(self) -> Fraction:
        return _frac((self.numerator, self.denominator))

    def _payload(self) -> dict:
        return {
            "world_axis": self.world_axis,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "contributors": [value.to_record() for value in self.contributors],
            "contributor_count": self.contributor_count,
        }

    def to_record(self) -> dict:
        record = self._payload()
        record["hypothesis_sha256"] = self.hypothesis_sha256
        return record


def _make_hypothesis(world_axis: int, value: Fraction,
                     contributors: Sequence[V2BAnchorContributor]
                     ) -> V2BAnchorHypothesis:
    ordered = tuple(contributors)
    payload = {
        "world_axis": world_axis,
        "numerator": value.numerator,
        "denominator": value.denominator,
        "contributors": [item.to_record() for item in ordered],
        "contributor_count": len(ordered),
    }
    return V2BAnchorHypothesis(
        world_axis=world_axis, numerator=value.numerator,
        denominator=value.denominator, contributors=ordered,
        contributor_count=len(ordered),
        hypothesis_sha256=_sha256(payload))


# ---------------------------------------------------------------------------
# Split residue binding and G-phase reconstruction
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class V2BSplitResidueBinding:
    """Frozen chain P0 result -> G evidence -> P1-B replay -> this binding."""

    split_phase_result_sha256: str
    generation_evidence_sha256: str
    g_reconstruction_sha256: str
    supplied_evidence_root_sha256: str
    replayed_evidence_root_sha256: str
    phase_candidates_sha256: tuple[str, ...]
    supplied_evidence_sha256s: tuple[str, ...]
    replayed_evidence_sha256s: tuple[str, ...]
    binding_sha256: str

    def __post_init__(self) -> None:
        for name in ("split_phase_result_sha256", "generation_evidence_sha256",
                     "g_reconstruction_sha256", "supplied_evidence_root_sha256",
                     "replayed_evidence_root_sha256"):
            if not _hex64(getattr(self, name)):
                raise ValueError("split_residue_binding_mismatch")
        for name in ("phase_candidates_sha256", "supplied_evidence_sha256s",
                     "replayed_evidence_sha256s"):
            value = getattr(self, name)
            if not isinstance(value, tuple) or any(not _hex64(item) for item in value):
                raise ValueError("split_residue_binding_mismatch")
        if (self.supplied_evidence_root_sha256
                != _root_sha256(self.supplied_evidence_sha256s)
                or self.replayed_evidence_root_sha256
                != _root_sha256(self.replayed_evidence_sha256s)
                or self.supplied_evidence_sha256s != self.replayed_evidence_sha256s):
            raise ValueError("p1b_source_mismatch")
        if (not _hex64(self.binding_sha256)
                or self.binding_sha256 != _sha256(self._payload())):
            raise ValueError("split_residue_binding_mismatch")

    def _payload(self) -> dict:
        return {
            "split_phase_result_sha256": self.split_phase_result_sha256,
            "generation_evidence_sha256": self.generation_evidence_sha256,
            "g_reconstruction_sha256": self.g_reconstruction_sha256,
            "supplied_evidence_root_sha256": self.supplied_evidence_root_sha256,
            "replayed_evidence_root_sha256": self.replayed_evidence_root_sha256,
            "phase_candidates_sha256": list(self.phase_candidates_sha256),
            "supplied_evidence_sha256s": list(self.supplied_evidence_sha256s),
            "replayed_evidence_sha256s": list(self.replayed_evidence_sha256s),
        }

    def to_record(self) -> dict:
        record = self._payload()
        record["binding_sha256"] = self.binding_sha256
        return record


def _root_sha256(values: Sequence[str]) -> str:
    return _sha256(list(values))


@dataclass(frozen=True, slots=True)
class V2BGPhaseReconstruction:
    """One reading's G-only phase reconstruction and exact replay binding."""

    reading_identity: dict
    phase_uv_pixels: tuple[float | None, float | None]
    response_magnitude_uv: tuple[float | None, float | None]
    profile_mass_uv: tuple[float | None, float | None]
    concentration_uv: tuple[float | None, float | None]
    unavailable_reason_uv: tuple[str | None, str | None]
    period_pixels: float
    patch_centre_uv: tuple[float, float]
    extent: float
    samples: int
    source_region_id: str | None
    partition_sha256: str
    G_u_sha256: str
    G_v_sha256: str
    g_profile_sha256: str
    accepted_reading_sha256: str
    reconstructed_reading_sha256: str
    reconstruction_sha256: str

    def __post_init__(self) -> None:
        identity = _split_reading_identity(self.reading_identity)
        if identity is None:
            raise ValueError("g_phase_reconstruction_refused")
        object.__setattr__(self, "reading_identity", identity)
        for name in ("phase_uv_pixels", "response_magnitude_uv",
                     "profile_mass_uv", "concentration_uv",
                     "unavailable_reason_uv"):
            value = getattr(self, name)
            if not isinstance(value, tuple) or len(value) != 2:
                raise ValueError("g_phase_reconstruction_refused")
        for name in ("period_pixels", "extent"):
            value = _finite_float(getattr(self, name))
            if value is None or value <= 0.0:
                raise ValueError("g_phase_reconstruction_refused")
            object.__setattr__(self, name, value)
        if (type(self.samples) is not int or self.samples < 2
                or not isinstance(self.patch_centre_uv, tuple)
                or len(self.patch_centre_uv) != 2):
            raise ValueError("g_phase_reconstruction_refused")
        if self.source_region_id is not None and not _hex64(self.source_region_id):
            raise ValueError("g_phase_reconstruction_refused")
        for name in ("partition_sha256", "G_u_sha256", "G_v_sha256",
                     "g_profile_sha256", "accepted_reading_sha256",
                     "reconstructed_reading_sha256"):
            if not _hex64(getattr(self, name)):
                raise ValueError("g_phase_reconstruction_refused")
        if (not _hex64(self.reconstruction_sha256)
                or self.reconstruction_sha256 != _sha256(self._payload())):
            raise ValueError("g_phase_reconstruction_refused")

    def _payload(self) -> dict:
        return {
            "reading_identity": dict(self.reading_identity),
            "phase_uv_pixels": list(self.phase_uv_pixels),
            "response_magnitude_uv": list(self.response_magnitude_uv),
            "profile_mass_uv": list(self.profile_mass_uv),
            "concentration_uv": list(self.concentration_uv),
            "unavailable_reason_uv": list(self.unavailable_reason_uv),
            "period_pixels": self.period_pixels,
            "patch_centre_uv": list(self.patch_centre_uv),
            "extent": self.extent,
            "samples": self.samples,
            "source_region_id": self.source_region_id,
            "partition_sha256": self.partition_sha256,
            "G_u_sha256": self.G_u_sha256,
            "G_v_sha256": self.G_v_sha256,
            "g_profile_sha256": self.g_profile_sha256,
            "accepted_reading_sha256": self.accepted_reading_sha256,
            "reconstructed_reading_sha256": self.reconstructed_reading_sha256,
        }

    def to_record(self) -> dict:
        record = self._payload()
        record["reconstruction_sha256"] = self.reconstruction_sha256
        return record


def _split_reading_identity(value: Any) -> dict | None:
    if (not isinstance(value, dict)
            or set(value) != {"frame", "plane_index", "reading_axis"}):
        return None
    if (type(value["frame"]) is not str or not value["frame"]
            or type(value["plane_index"]) is not int or value["plane_index"] < 0
            or type(value["reading_axis"]) is not int
            or value["reading_axis"] not in _WORLD_AXES):
        return None
    return {"frame": value["frame"], "plane_index": value["plane_index"],
            "reading_axis": value["reading_axis"]}


# ---------------------------------------------------------------------------
# Observations, axis results, prediction, result, consistency diagnostic
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class V2BAnchorObservation:
    """One hypothesis tested against one held-out D observation."""

    reading_identity: dict
    local_axis: str
    world_axis: int
    world_sign: int
    selected_b: tuple[int, int]
    partition_sha256: str
    G_a_sha256: str
    D_a_sha256: str
    generation_evidence_sha256: str
    seam_evidence_sha256: str
    d_profile_sha256: str
    g_reconstruction_sha256: str
    split_residue_binding_sha256: str
    boundary_energy: tuple[int, int] | None
    interior_energy: tuple[int, int] | None
    margin: tuple[int, int] | None
    zero_coverage: bool
    held_out_samples: int
    hypothesis_sha256: str
    propagation_origin: tuple[int, int]
    observation_sha256: str

    def __post_init__(self) -> None:
        identity = _split_reading_identity(self.reading_identity)
        if identity is None:
            raise ValueError("v2b_input_invalid")
        object.__setattr__(self, "reading_identity", identity)
        if self.local_axis not in _LOCAL_AXES \
                or type(self.world_axis) is not int or self.world_axis not in _WORLD_AXES \
                or self.world_sign not in (-1, 1):
            raise ValueError("axis_transform_invalid")
        if not _validate_reduced(self.selected_b):
            raise ValueError("noncanonical_dyadic")
        if not _validate_dyadic(self.propagation_origin):
            raise ValueError("noncanonical_dyadic")
        for name in ("partition_sha256", "G_a_sha256", "D_a_sha256",
                     "generation_evidence_sha256", "seam_evidence_sha256",
                     "d_profile_sha256", "g_reconstruction_sha256",
                     "split_residue_binding_sha256", "hypothesis_sha256"):
            if not _hex64(getattr(self, name)):
                raise ValueError("v2b_input_invalid")
        if type(self.zero_coverage) is not bool:
            raise ValueError("v2b_input_invalid")
        if type(self.held_out_samples) is not int or self.held_out_samples < 0:
            raise ValueError("v2b_input_invalid")
        if self.zero_coverage:
            if any(getattr(self, name) is not None
                   for name in ("boundary_energy", "interior_energy", "margin")):
                raise ValueError("v2b_input_invalid")
        else:
            for name in ("boundary_energy", "interior_energy", "margin"):
                if not _validate_reduced(getattr(self, name)):
                    raise ValueError("v2b_input_invalid")
        if (not _hex64(self.observation_sha256)
                or self.observation_sha256 != _sha256(self._payload())):
            raise ValueError("v2b_input_invalid")

    def _payload(self) -> dict:
        return {
            "reading_identity": dict(self.reading_identity),
            "local_axis": self.local_axis,
            "world_axis": self.world_axis,
            "world_sign": self.world_sign,
            "selected_b": list(self.selected_b),
            "partition_sha256": self.partition_sha256,
            "G_a_sha256": self.G_a_sha256,
            "D_a_sha256": self.D_a_sha256,
            "generation_evidence_sha256": self.generation_evidence_sha256,
            "seam_evidence_sha256": self.seam_evidence_sha256,
            "d_profile_sha256": self.d_profile_sha256,
            "g_reconstruction_sha256": self.g_reconstruction_sha256,
            "split_residue_binding_sha256": self.split_residue_binding_sha256,
            "boundary_energy": (None if self.boundary_energy is None
                                else list(self.boundary_energy)),
            "interior_energy": (None if self.interior_energy is None
                                else list(self.interior_energy)),
            "margin": None if self.margin is None else list(self.margin),
            "zero_coverage": self.zero_coverage,
            "held_out_samples": self.held_out_samples,
            "hypothesis_sha256": self.hypothesis_sha256,
            "propagation_origin": list(self.propagation_origin),
        }

    def to_record(self) -> dict:
        record = self._payload()
        record["observation_sha256"] = self.observation_sha256
        return record


@dataclass(frozen=True, slots=True)
class V2BAnchorAxisResult:
    """One world axis' determination outcome."""

    world_axis: int
    status: str
    alpha_dyadic: tuple[int, int] | None
    reason: str | None
    accepted_hypotheses: tuple[V2BAnchorHypothesis, ...]
    rejected_hypotheses: tuple[V2BAnchorHypothesis, ...]
    distinct_frame_count: int
    axis_sha256: str

    def __post_init__(self) -> None:
        if type(self.world_axis) is not int or self.world_axis not in _WORLD_AXES \
                or self.status not in ("determined", "abstained"):
            raise ValueError("v2b_input_invalid")
        if self.status == "determined":
            if (self.alpha_dyadic is None or self.reason is not None
                    or len(self.accepted_hypotheses) != 1):
                raise ValueError("v2b_input_invalid")
            if not _validate_dyadic(self.alpha_dyadic):
                raise ValueError("noncanonical_dyadic")
        else:
            if self.alpha_dyadic is not None or self.reason not in V2B_ABSTENTION_REASONS:
                raise ValueError("v2b_input_invalid")
        for value in (*self.accepted_hypotheses, *self.rejected_hypotheses):
            if not isinstance(value, V2BAnchorHypothesis) \
                    or value.world_axis != self.world_axis:
                raise ValueError("v2b_input_invalid")
        if type(self.distinct_frame_count) is not int or self.distinct_frame_count < 0:
            raise ValueError("v2b_input_invalid")
        if (not _hex64(self.axis_sha256)
                or self.axis_sha256 != _sha256(self._payload())):
            raise ValueError("v2b_input_invalid")

    @property
    def alpha_fraction(self) -> Fraction | None:
        return None if self.alpha_dyadic is None else _frac(self.alpha_dyadic)

    def _payload(self) -> dict:
        return {
            "world_axis": self.world_axis,
            "status": self.status,
            "alpha_dyadic": (None if self.alpha_dyadic is None
                             else list(self.alpha_dyadic)),
            "reason": self.reason,
            "accepted_hypotheses": [h.hypothesis_sha256
                                    for h in self.accepted_hypotheses],
            "rejected_hypotheses": [h.hypothesis_sha256
                                    for h in self.rejected_hypotheses],
            "distinct_frame_count": self.distinct_frame_count,
        }

    def to_record(self) -> dict:
        record = self._payload()
        record["axis_sha256"] = self.axis_sha256
        return record


# ---------------------------------------------------------------------------
# Retained-record binding for work counters
# ---------------------------------------------------------------------------

def _observation_accepts(observation: "V2BAnchorObservation",
                         calibration: V2BAnchorCalibration) -> bool:
    """Re-run the frozen absolute test from a retained observation record."""
    if observation.zero_coverage:
        return False
    margin = _frac(observation.margin)
    boundary = _frac(observation.boundary_energy)
    return (margin >= calibration.tau_margin_fraction
            and boundary >= calibration.tau_energy_fraction
            and observation.held_out_samples
            >= calibration.min_held_out_samples)


def _derived_accepted_frame_supports(
        observations: Sequence["V2BAnchorObservation"],
        calibration: V2BAnchorCalibration) -> int:
    """``K``: unique ``(axis, hypothesis, frame)`` triples accepted at least once."""
    triples = set()
    for observation in observations:
        if _observation_accepts(observation, calibration):
            triples.add((observation.world_axis, observation.hypothesis_sha256,
                         observation.reading_identity["frame"]))
    return len(triples)


def _derived_upper_bound(hypotheses: Sequence["V2BAnchorHypothesis"],
                         observations: Sequence["V2BAnchorObservation"]) -> int:
    """``U = sum_w |H_w| * |{frame(o): o in Q_w}|`` from retained records."""
    per_axis = {axis: 0 for axis in _WORLD_AXES}
    for hypothesis in hypotheses:
        per_axis[hypothesis.world_axis] += 1
    frames = {axis: set() for axis in _WORLD_AXES}
    for observation in observations:
        frames[observation.world_axis].add(
            observation.reading_identity["frame"])
    return sum(per_axis[axis] * len(frames[axis]) for axis in _WORLD_AXES)


def _derived_held_out_bounds(
        observations: Sequence["V2BAnchorObservation"]) -> tuple[int, int]:
    """Bounds on the per-reading distinct ordered D-coordinate union maximum.

    The retained observations only carry each axis' own ``|D_a|`` coverage, so
    the exact union cardinality is not recoverable from them.  The union of two
    axis sets is at least the larger and at most their sum, which is enough to
    reject a freshly resealed held-out maximum that cannot come from any source
    with these coverage counts.
    """
    per_reading: dict[tuple, dict[int, int]] = {}
    for observation in observations:
        identity = observation.reading_identity
        key = (identity["frame"], identity["plane_index"],
               identity["reading_axis"])
        coverage = per_reading.setdefault(key, {})
        existing = coverage.get(observation.world_axis)
        if existing is not None and existing != observation.held_out_samples:
            raise ValueError("v2b_input_invalid")
        coverage[observation.world_axis] = observation.held_out_samples
    lower = 0
    upper = 0
    for coverage in per_reading.values():
        values = list(coverage.values())
        lower = max(lower, max(values))
        upper = max(upper, sum(values))
    return lower, upper


@dataclass(frozen=True, slots=True)
class V2BAnchorPrediction:
    """The immutable V2-B determination record."""

    status: str
    reason: str
    alpha_x: tuple[int, int] | None
    alpha_y: tuple[int, int] | None
    alpha_z: tuple[int, int] | None
    axes: tuple[V2BAnchorAxisResult, ...]
    c2rb_result_sha256: str
    c2rb_input_sha256: str
    vector_result_sha256: str
    v2a_result_sha256: str
    v2a_input_sha256: str
    kernel_result_sha256: str
    canonical_universe_sha256: str
    split_phase_result_sha256: str
    split_phase_result_sha256s: tuple[str, ...]
    split_generation_evidence_sha256: str
    split_g_phase_reconstruction_sha256: str
    split_seam_profile_evidence_sha256: str
    split_residue_binding_sha256: str
    p1b_hypothesis_phase_shift_evidence_sha256: str
    v2a_replay_ok: bool
    hypotheses: tuple[V2BAnchorHypothesis, ...]
    observations: tuple[V2BAnchorObservation, ...]
    planned_work: V2BAnchorWork
    runtime_work: V2BAnchorWork
    refusals: tuple[V2BAnchorRefusal, ...]
    calibration_sha256: str
    budget_sha256: str
    input_sha256: str
    result_sha256: str

    def __post_init__(self) -> None:
        if self.status not in ("determined", "partial", "abstained") \
                or self.reason not in V2B_PREDICTION_REASONS:
            raise ValueError("v2b_input_invalid")
        if not isinstance(self.axes, tuple) \
                or any(not isinstance(value, V2BAnchorAxisResult)
                       for value in self.axes) \
                or tuple(value.world_axis for value in self.axes) != _WORLD_AXES:
            raise ValueError("v2b_input_invalid")
        determined = [value.status == "determined" for value in self.axes]
        expected = ("determined" if all(determined) else
                    "abstained" if not any(determined) else "partial")
        if self.status != expected:
            raise ValueError("v2b_input_invalid")
        alphas = (self.alpha_x, self.alpha_y, self.alpha_z)
        for axis, alpha in zip(self.axes, alphas):
            if axis.status == "determined":
                if alpha != axis.alpha_dyadic or not _validate_dyadic(alpha):
                    raise ValueError("v2b_input_invalid")
            elif alpha is not None:
                raise ValueError("v2b_input_invalid")
        if type(self.v2a_replay_ok) is not bool:
            raise ValueError("v2b_input_invalid")
        for name in ("c2rb_result_sha256", "c2rb_input_sha256",
                     "vector_result_sha256", "v2a_result_sha256",
                     "v2a_input_sha256", "kernel_result_sha256",
                     "canonical_universe_sha256", "split_phase_result_sha256",
                     "split_generation_evidence_sha256",
                     "split_g_phase_reconstruction_sha256",
                     "split_seam_profile_evidence_sha256",
                     "split_residue_binding_sha256",
                     "p1b_hypothesis_phase_shift_evidence_sha256",
                     "calibration_sha256", "budget_sha256",
                     "input_sha256", "result_sha256"):
            if not _hex64(getattr(self, name)):
                raise ValueError("v2b_input_invalid")
        if not isinstance(self.planned_work, V2BAnchorWork) \
                or not isinstance(self.runtime_work, V2BAnchorWork) \
                or not isinstance(self.refusals, tuple) \
                or any(not isinstance(value, V2BAnchorRefusal)
                       for value in self.refusals):
            raise ValueError("v2b_input_invalid")
        for value in self.hypotheses:
            if not isinstance(value, V2BAnchorHypothesis):
                raise ValueError("v2b_input_invalid")
        for value in self.observations:
            if not isinstance(value, V2BAnchorObservation):
                raise ValueError("v2b_input_invalid")
        self._validate_retained_binding()
        if self.input_sha256 != _sha256({
                "c2rb_result_sha256": self.c2rb_result_sha256,
                "vector_result_sha256": self.vector_result_sha256,
                "v2a_result_sha256": self.v2a_result_sha256,
                "split_phase_result_sha256": self.split_phase_result_sha256,
                "split_residue_binding_sha256": self.split_residue_binding_sha256,
                "calibration_sha256": self.calibration_sha256,
                "budget_sha256": self.budget_sha256}):
            raise ValueError("result_validation_mismatch")
        if (not _hex64(self.result_sha256)
                or self.result_sha256 != _sha256(self._payload())):
            raise ValueError("result_validation_mismatch")

    def _validate_retained_binding(self) -> None:
        """Bind both work snapshots to the retained hypotheses and observations.

        A freshly resealed prediction recomputes its own digests, so only the
        independently retained records can falsify a contradictory counter: the
        hypothesis/observation relation, the contributor count sums, the planned
        upper bound ``U`` and the accepted frame-support count ``K`` are all
        re-derived here and must match the reported snapshots exactly.
        """
        per_axis = {axis: 0 for axis in _WORLD_AXES}
        axis_keys: set[tuple[int, str]] = set()
        for axis_result in self.axes:
            per_axis[axis_result.world_axis] = (
                len(axis_result.accepted_hypotheses)
                + len(axis_result.rejected_hypotheses))
            for hypothesis in (*axis_result.accepted_hypotheses,
                               *axis_result.rejected_hypotheses):
                axis_keys.add((axis_result.world_axis,
                               hypothesis.hypothesis_sha256))
        retained: set[tuple[int, str]] = set()
        for hypothesis in self.hypotheses:
            key = (hypothesis.world_axis, hypothesis.hypothesis_sha256)
            if key in retained:
                raise ValueError("v2b_input_invalid")
            retained.add(key)
        if retained != axis_keys:
            raise ValueError("v2b_input_invalid")
        contributor_per = max(
            (value.contributor_count for value in self.hypotheses), default=0)
        contributor_total = sum(
            value.contributor_count for value in self.hypotheses)
        for work in (self.planned_work, self.runtime_work):
            if (work.hypotheses_per_axis != max(per_axis.values(), default=0)
                    or work.hypotheses_total != sum(per_axis.values())
                    or work.contributor_refs_per_hypothesis != contributor_per
                    or work.contributor_refs_total != contributor_total):
                raise ValueError("v2b_input_invalid")
        for observation in self.observations:
            if (observation.world_axis, observation.hypothesis_sha256) \
                    not in retained:
                raise ValueError("v2b_input_invalid")
        runtime = self.runtime_work
        if runtime.observations != len(self.observations):
            raise ValueError("v2b_input_invalid")
        upper = _derived_upper_bound(self.hypotheses, self.observations)
        accepted = _derived_accepted_frame_supports(
            self.observations, V2B_BUILTIN_CALIBRATION)
        if self.planned_work.accepted_frame_supports != upper:
            raise ValueError("v2b_input_invalid")
        if runtime.accepted_frame_supports != accepted:
            raise ValueError("v2b_input_invalid")
        if accepted > upper:
            raise ValueError("v2b_input_invalid")
        if runtime.accepted_frame_supports > runtime.observations:
            raise ValueError("v2b_input_invalid")
        if self.planned_work.observations < runtime.observations:
            raise ValueError("v2b_input_invalid")
        for name in _PREDICTION_SHARED_FIELDS:
            if getattr(self.planned_work, name) != getattr(runtime, name):
                raise ValueError("v2b_input_invalid")
        lower, high = _derived_held_out_bounds(self.observations)
        if not lower <= runtime.held_out_samples_per_reading <= high:
            raise ValueError("v2b_input_invalid")

    def _payload(self) -> dict:
        return {
            "schema": V2B_ANCHOR_SCHEMA,
            "status": self.status,
            "reason": self.reason,
            "alpha_x": None if self.alpha_x is None else list(self.alpha_x),
            "alpha_y": None if self.alpha_y is None else list(self.alpha_y),
            "alpha_z": None if self.alpha_z is None else list(self.alpha_z),
            "axes": [value.to_record() for value in self.axes],
            "c2rb_result_sha256": self.c2rb_result_sha256,
            "c2rb_input_sha256": self.c2rb_input_sha256,
            "vector_result_sha256": self.vector_result_sha256,
            "v2a_result_sha256": self.v2a_result_sha256,
            "v2a_input_sha256": self.v2a_input_sha256,
            "kernel_result_sha256": self.kernel_result_sha256,
            "canonical_universe_sha256": self.canonical_universe_sha256,
            "split_phase_result_sha256": self.split_phase_result_sha256,
            "split_phase_result_sha256s": list(self.split_phase_result_sha256s),
            "split_generation_evidence_sha256": self.split_generation_evidence_sha256,
            "split_g_phase_reconstruction_sha256": self.split_g_phase_reconstruction_sha256,
            "split_seam_profile_evidence_sha256": self.split_seam_profile_evidence_sha256,
            "split_residue_binding_sha256": self.split_residue_binding_sha256,
            "p1b_hypothesis_phase_shift_evidence_sha256":
                self.p1b_hypothesis_phase_shift_evidence_sha256,
            "v2a_replay_ok": self.v2a_replay_ok,
            "hypotheses": [value.to_record() for value in self.hypotheses],
            "observations": [value.to_record() for value in self.observations],
            "planned_work": self.planned_work.to_record(),
            "runtime_work": self.runtime_work.to_record(),
            "refusals": [value.to_record() for value in self.refusals],
            "calibration_sha256": self.calibration_sha256,
            "budget_sha256": self.budget_sha256,
            "input_sha256": self.input_sha256,
        }

    def to_record(self) -> dict:
        record = self._payload()
        record["result_sha256"] = self.result_sha256
        return record

    def to_json_bytes(self) -> bytes:
        return V2B_CANONICAL_JSON(self.to_record())

    @classmethod
    def build(cls, **fields: Any) -> "V2BAnchorPrediction":
        if "input_sha256" not in fields:
            fields["input_sha256"] = _sha256({
                "c2rb_result_sha256": fields["c2rb_result_sha256"],
                "vector_result_sha256": fields["vector_result_sha256"],
                "v2a_result_sha256": fields["v2a_result_sha256"],
                "split_phase_result_sha256": fields["split_phase_result_sha256"],
                "split_residue_binding_sha256":
                    fields["split_residue_binding_sha256"],
                "calibration_sha256": fields["calibration_sha256"],
                "budget_sha256": fields["budget_sha256"]})
        provisional = object.__new__(cls)
        for name, value in fields.items():
            object.__setattr__(provisional, name, value)
        object.__setattr__(provisional, "result_sha256", "")
        payload = provisional.to_record()
        payload.pop("result_sha256")
        object.__setattr__(provisional, "result_sha256", _sha256(payload))
        cls.__post_init__(provisional)
        return provisional


@dataclass(frozen=True, slots=True)
class V2BAnchorResult:
    """Atomic union of prediction, refusal, or budget outcome."""

    status: str
    prediction: V2BAnchorPrediction | None
    refusals: tuple[V2BAnchorRefusal, ...]
    budget_stage: str | None
    work: V2BAnchorWork | None
    result_sha256: str

    def __post_init__(self) -> None:
        if self.status not in ("determined", "partial", "abstained",
                               "refused", "budget_exhausted"):
            raise ValueError("v2b_input_invalid")
        if self.budget_stage not in (None, "planned", "runtime"):
            raise ValueError("budget_exhausted")
        if not isinstance(self.refusals, tuple) \
                or any(not isinstance(value, V2BAnchorRefusal)
                       for value in self.refusals):
            raise ValueError("v2b_input_invalid")
        if self.work is not None and not isinstance(self.work, V2BAnchorWork):
            raise ValueError("v2b_input_invalid")
        if self.status in ("determined", "partial", "abstained"):
            if (not isinstance(self.prediction, V2BAnchorPrediction)
                    or self.prediction.status != self.status
                    or self.refusals or self.budget_stage is not None):
                raise ValueError("result_validation_mismatch")
            # The outer snapshot is the prediction's runtime snapshot; a stale
            # outer copy that disagrees is never accepted.
            if (self.work is None
                    or self.work.to_record()
                    != self.prediction.runtime_work.to_record()):
                raise ValueError("result_validation_mismatch")
        elif self.status == "refused":
            if (self.prediction is not None or not self.refusals
                    or self.budget_stage is not None):
                raise ValueError("v2b_input_invalid")
        else:
            if (self.prediction is not None or len(self.refusals) != 1
                    or self.refusals[0].reason != "budget_exhausted"
                    or self.budget_stage not in ("planned", "runtime")):
                raise ValueError("budget_exhausted")
            self._validate_budget_branch()
        if (not _hex64(self.result_sha256)
                or self.result_sha256 != _sha256(self._payload())):
            raise ValueError("result_validation_mismatch")

    def _validate_budget_branch(self) -> None:
        """Bind an atomic exhaustion branch to its exact snapshots and aliases.

        Planned exhaustion keeps the exact planned snapshot and no runtime
        snapshot, and may only name a static preflight cap.  Runtime exhaustion
        keeps the exact runtime snapshot, the exact planned snapshot in the
        bounded detail, matching stage/cap/observed aliases, and -- for the
        amended support unit -- ``K == limit + 1 <= U``.  Every refusal limit
        must be an exact integer of at least one.  Neither branch may carry
        prediction, hypothesis, observation, or anchor evidence.
        """
        if self.work is None or not isinstance(self.work, V2BAnchorWork):
            raise ValueError("budget_exhausted")
        detail = self.refusals[0].detail
        if not isinstance(detail, dict) or set(detail) - _BUDGET_DETAIL_KEYS:
            raise ValueError("budget_exhausted")
        if detail.get("budget_stage") != self.budget_stage:
            raise ValueError("budget_exhausted")
        limit = detail.get("limit")
        observed = detail.get("observed")
        # Every public cap is a positive exact integer, so a refusal limit below
        # one is never a real stop.  This is checked before any branch-specific
        # validation so a ``limit = 0`` record is refused as a limit, not as a
        # coincidental alias mismatch.
        if (type(limit) is not int or type(observed) is not int
                or limit < 1 or observed <= limit):
            raise ValueError("budget_exhausted")
        cap = detail.get("cap")
        if cap not in _BUDGET_FIELDS:
            raise ValueError("budget_exhausted")
        # Branch-specific cap eligibility: a planned exhaustion can only name a
        # static preflight counter that ``_first_over_cap`` actually evaluates.
        # ``max_accepted_frame_supports`` is enforced only at runtime insertion,
        # so it can never be the cap of a planned branch.
        if self.budget_stage == "planned" \
                and cap not in _PLANNED_EXHAUSTIBLE_CAPS:
            raise ValueError("budget_exhausted")
        field = _CAP_TO_WORK[cap]
        if getattr(self.work, field) != observed:
            raise ValueError("budget_exhausted")
        if self.budget_stage == "planned":
            if "runtime_work" in detail or "planned_work" in detail:
                raise ValueError("budget_exhausted")
            return
        planned = detail.get("planned_work")
        runtime = detail.get("runtime_work")
        if not isinstance(planned, dict) or not isinstance(runtime, dict):
            raise ValueError("budget_exhausted")
        try:
            planned_work = V2BAnchorWork(**planned)
            runtime_work = V2BAnchorWork(**runtime)
        except (TypeError, ValueError):
            raise ValueError("budget_exhausted")
        if runtime_work.to_record() != self.work.to_record():
            raise ValueError("budget_exhausted")
        if getattr(runtime_work, field) != observed:
            raise ValueError("budget_exhausted")
        if runtime_work.accepted_frame_supports > runtime_work.observations:
            raise ValueError("budget_exhausted")
        if runtime_work.observations > planned_work.observations:
            raise ValueError("budget_exhausted")
        if (runtime_work.accepted_frame_supports
                > planned_work.accepted_frame_supports):
            raise ValueError("budget_exhausted")
        if (runtime_work.held_out_samples_per_reading
                > planned_work.held_out_samples_per_reading):
            raise ValueError("budget_exhausted")
        for name in _PLANNED_RUNTIME_SHARED_FIELDS:
            if getattr(runtime_work, name) != getattr(planned_work, name):
                raise ValueError("budget_exhausted")
        if cap == "max_accepted_frame_supports" and observed != limit + 1:
            raise ValueError("budget_exhausted")

    def _payload(self) -> dict:
        return {
            "status": self.status,
            "prediction": (None if self.prediction is None
                           else self.prediction.to_record()),
            "refusals": [value.to_record() for value in self.refusals],
            "budget_stage": self.budget_stage,
            "work": None if self.work is None else self.work.to_record(),
        }

    def to_record(self) -> dict:
        record = self._payload()
        record["result_sha256"] = self.result_sha256
        return record

    def to_json_bytes(self) -> bytes:
        return V2B_CANONICAL_JSON(self.to_record())


def _make_result(**fields: Any) -> V2BAnchorResult:
    provisional = object.__new__(V2BAnchorResult)
    for name, value in fields.items():
        object.__setattr__(provisional, name, value)
    object.__setattr__(provisional, "result_sha256", "")
    payload = provisional.to_record()
    payload.pop("result_sha256")
    object.__setattr__(provisional, "result_sha256", _sha256(payload))
    V2BAnchorResult.__post_init__(provisional)
    return provisional


@dataclass(frozen=True, slots=True)
class V2BConsistencyDiagnostic:
    """Path A-only, non-promotable consistency diagnostic (never ``alpha``)."""

    status: str
    axes: tuple[dict, ...]
    period_pixels_binding: tuple[float, ...]
    full_patch_evidence_root_sha256: str
    diagnostic_sha256: str

    def __post_init__(self) -> None:
        if self.status not in ("consistent", "inconsistent", "unavailable",
                               "insufficient_data"):
            raise ValueError("v2b_input_invalid")
        if (not isinstance(self.axes, tuple)
                or any(not isinstance(value, dict) for value in self.axes)
                or tuple(value.get("world_axis") for value in self.axes)
                != _WORLD_AXES):
            raise ValueError("v2b_input_invalid")
        for value in self.axes:
            if value.get("status") not in (
                    "consistent", "inconsistent", "unavailable",
                    "insufficient_data"):
                raise ValueError("v2b_input_invalid")
        if not isinstance(self.period_pixels_binding, tuple) \
                or any(_finite_float(item) is None
                       for item in self.period_pixels_binding):
            raise ValueError("v2b_input_invalid")
        if not _hex64(self.full_patch_evidence_root_sha256):
            raise ValueError("v2b_input_invalid")
        if (not _hex64(self.diagnostic_sha256)
                or self.diagnostic_sha256 != _sha256(self._payload())):
            raise ValueError("v2b_input_invalid")

    def _payload(self) -> dict:
        return {
            "status": self.status,
            "axes": [dict(value) for value in self.axes],
            "period_pixels_binding": [float(value)
                                      for value in self.period_pixels_binding],
            "full_patch_evidence_root_sha256": self.full_patch_evidence_root_sha256,
        }

    def to_record(self) -> dict:
        record = self._payload()
        record["diagnostic_sha256"] = self.diagnostic_sha256
        return record


# ---------------------------------------------------------------------------
# Internal helpers: reading lookup, seam profiles, propagation
# ---------------------------------------------------------------------------

def _ordered_readings(readings: Sequence[lattice_planes.PlaneReading]
                      ) -> dict[str, tuple[lattice_planes.PlaneReading, ...]]:
    by_frame: dict[str, list[lattice_planes.PlaneReading]] = {}
    for reading in readings:
        if not isinstance(reading, lattice_planes.PlaneReading):
            raise ValueError("v2b_input_invalid")
        by_frame.setdefault(reading.frame, []).append(reading)
    return {frame: tuple(sorted(values, key=lattice_planes.reading_order_key))
            for frame, values in by_frame.items()}


def _reading_position(ordered: Mapping[str, tuple], frame: str,
                      plane_index: int, axis: int) -> lattice_planes.PlaneReading:
    values = ordered.get(frame)
    if not values or not 0 <= plane_index < len(values):
        raise ValueError("g_phase_reconstruction_refused")
    reading = values[plane_index]
    if reading.frame != frame or reading.axis != axis:
        raise ValueError("g_phase_reconstruction_refused")
    return reading


def _reading_record(reading: lattice_planes.PlaneReading) -> dict:
    rectification = reading.rectification
    evidence = reading.phase_evidence
    return {
        "frame": reading.frame,
        "axis": reading.axis,
        "side": float(reading.side),
        "alternatives": sorted(
            [{"b": float(value.blocks_per_period),
              "distance": float(value.distance_blocks),
              "support": float(value.support)}
             for value in reading.alternatives],
            key=lambda item: (item["b"], item["distance"], item["support"])),
        "period_pixels": float(reading.period_pixels),
        "phase_pixels": [float(value) for value in reading.phase_pixels],
        "extent": float(reading.extent),
        "samples": int(reading.samples),
        "coverage_cells": int(reading.coverage_cells),
        "structural_share": float(reading.structural_share),
        "confidence": float(reading.confidence),
        "rectification": {
            "axis_matrix": [list(row) for row in rectification.axis_matrix],
            "focal": float(rectification.focal),
            "principal_point": list(rectification.principal_point),
            "centre": list(rectification.centre),
        },
        "phase_evidence": None if evidence is None else {
            "phase_uv_pixels": list(evidence.phase_uv_pixels),
            "period_pixels": float(evidence.period_pixels),
            "patch_centre_uv": list(evidence.patch_centre_uv),
            "extent": float(evidence.extent),
            "samples": int(evidence.samples),
            "response_magnitude_uv": list(evidence.response_magnitude_uv),
            "profile_mass_uv": list(evidence.profile_mass_uv),
            "concentration_uv": list(evidence.concentration_uv),
            "unavailable_reason_uv": list(evidence.unavailable_reason_uv),
            "source_region_id": evidence.source_region_id,
        },
    }


def _reading_sha256(reading: lattice_planes.PlaneReading) -> str:
    return _sha256(_reading_record(reading))


def _decode_seam_profile(record: lattice_planes.SplitSeamProfileEvidence,
                         height: int, width: int
                         ) -> tuple[list[float], list[int]]:
    """Decode, re-hash and reconstruct one embedded held-out D profile."""
    if not isinstance(record, lattice_planes.SplitSeamProfileEvidence):
        raise ValueError("seam_profile_not_reconstructible")
    expected_count = width if record.local_axis == "u" else height
    if record.sample_count != expected_count:
        raise ValueError("seam_profile_payload_invalid")
    try:
        samples_raw = base64.b64decode(
            record.profile_samples_float64_le_base64, validate=True)
        weights_raw = base64.b64decode(
            record.profile_weights_float64_le_base64, validate=True)
        indices_raw = base64.b64decode(
            record.axis_bin_indices_int64_le_base64, validate=True)
    except (TypeError, ValueError) as error:
        raise ValueError("seam_profile_payload_invalid") from error
    blobs = []
    for raw in (samples_raw, weights_raw, indices_raw):
        blobs.append(len(raw).to_bytes(8, "little", signed=False))
        blobs.append(raw)
    expected_payload = hashlib.sha256(
        lattice_planes.SPLIT_SEAM_PROFILE_PAYLOAD_VERSION
        + b"".join(blobs)).hexdigest()
    if expected_payload != record.payload_sha256:
        raise ValueError("seam_profile_payload_invalid")
    samples = np.frombuffer(samples_raw, dtype="<f8")
    weights = np.frombuffer(weights_raw, dtype="<f8")
    indices = np.frombuffer(indices_raw, dtype="<i8")
    if (len(samples) != expected_count or len(weights) != expected_count
            or len(indices) != expected_count):
        raise ValueError("seam_profile_payload_invalid")
    if not np.array_equal(indices, np.arange(expected_count, dtype="<i8")):
        raise ValueError("seam_profile_payload_invalid")
    if not bool(np.all(np.isfinite(samples))) \
            or not bool(np.all(np.isfinite(weights))) \
            or bool(np.any(weights < 0.0)):
        raise ValueError("seam_profile_payload_invalid")
    return [float(value) for value in samples], [int(value) for value in weights]


def _generation_phase_evidence(
        generation: lattice_planes.SplitGenerationPhaseEvidence,
) -> lattice_planes.PeriodicEnergyPhaseEvidence:
    return lattice_planes.PeriodicEnergyPhaseEvidence(
        phase_uv_pixels=tuple(generation.phase_uv_pixels),
        period_pixels=generation.period_pixels,
        patch_centre_uv=tuple(generation.patch_centre_uv),
        extent=generation.extent,
        samples=generation.samples,
        response_magnitude_uv=tuple(generation.response_magnitude_uv),
        profile_mass_uv=tuple(generation.profile_mass_uv),
        concentration_uv=tuple(generation.concentration_uv),
        unavailable_reason_uv=tuple(generation.unavailable_reason_uv),
        source_region_id=generation.source_region_id)


def _phase_pixels_from_evidence(
        evidence: lattice_planes.PeriodicEnergyPhaseEvidence,
) -> tuple[float, float]:
    return tuple(float(value if value is not None else 0.0)
                 for value in evidence.phase_uv_pixels)


def _p1b_record_sha256(record: plane_residue.PeriodicResidueCandidates) -> str:
    return _sha256(record.to_record())


def _hypothesis_evidence_sha256(
        record: plane_residue.HypothesisPhaseShiftEvidence) -> str:
    return _sha256(record.to_record())


def _certificate_of(c2rb_result: plane_relative_translation.C2RBResult):
    component = c2rb_result.components[0]
    if len(component.translation_classes) != 1:
        raise ValueError("translation_class_not_unique")
    return component, component.translation_classes[0]


def _translations_by_frame(certificate) -> dict[str, tuple[Fraction, Fraction, Fraction]]:
    """Exact per-frame translations from the C2-R-B certificate.

    The certificate carries the full signed dyadic domain (integers and negative
    fractions included); only the periodic anchor consumes them modulo one.
    """
    translations: dict[str, tuple[Fraction, Fraction, Fraction]] = {}
    for frame, xyz in certificate.translations_dyadic_xyz:
        translations[frame] = tuple(_signed_dyadic(value) for value in xyz)
    return translations


# ---------------------------------------------------------------------------
# Hypothesis derivation
# ---------------------------------------------------------------------------

def _reading_tangent_label(reading_axis: int, local_axis: str) -> int:
    """Map a reading's local ``u``/``v`` axis to its shared sorted tangent label.

    The shared discrete convention (``plane_vector_domain``,
    ``plane_axis_support`` and ``plane_graph``) labels the two tangent axes with
    the world labels other than the reading's normal, sorted ascending, with
    ``u`` first and ``v`` second.  A P1-B record carries only the linear
    ``u``/``v`` name, so its contributor label must be derived from this
    convention rather than from the linear index; ``plane_graph.axis_transform``
    is then applied to the label, exactly as for every other contributor.
    """
    if (type(reading_axis) is not int or reading_axis not in _WORLD_AXES
            or local_axis not in _LOCAL_AXES):
        raise ValueError("axis_transform_invalid")
    tangents = tuple(label for label in _WORLD_AXES if label != reading_axis)
    return tangents[0 if local_axis == "u" else 1]


def _derive_hypotheses(
        c2rb_result, v2a_result, replayed_evidence,
        ordered: Mapping[str, tuple], budget: V2BAnchorBudget,
) -> tuple[dict[int, list[V2BAnchorHypothesis]], dict[str, str]]:
    """Build ``H_w`` with complete ordered contributor provenance (§4)."""
    component, certificate = _certificate_of(c2rb_result)
    translations = _translations_by_frame(certificate)
    contribs: dict[int, dict[Fraction, list[V2BAnchorContributor]]] = {
        axis: {} for axis in _WORLD_AXES}
    failures: dict[str, str] = {}
    order_counter = {axis: 0 for axis in _WORLD_AXES}

    def add(axis: int, value: Fraction, contributor: V2BAnchorContributor) -> None:
        bucket = contribs[axis].setdefault(_mod1(value), [])
        bucket.append(contributor)

    # 1. Accepted V2-A reading-domain residue candidates.
    for domain in v2a_result.reading_domains:
        identity = domain.reading_identity
        frame = identity.frame
        translation = translations.get(frame)
        if translation is None:
            failures[f"{frame}:{identity.canonical_index}"] = "propagation_unavailable"
            continue
        for local_axis, residue in (("u", domain.residue_u), ("v", domain.residue_v)):
            local_label = residue.local_axis_label
            try:
                world_axis, world_sign = plane_graph.axis_transform(
                    local_label, domain.absolute_quadrant)
            except ValueError:
                failures[f"{frame}:{identity.canonical_index}:{local_axis}"] = \
                    "axis_transform_invalid"
                continue
            world_translation = _reduce_dyadic(translation[world_axis])
            selected_b = _reduce_fraction(_fraction_from_float(residue.selected_b))
            for index, raw in enumerate(residue.candidate_values):
                value = _fraction_from_float(raw)
                anchor = _mod1(world_sign * value + translation[world_axis])
                contributor = _make_contributor(
                    source_kind="reading_domain",
                    source_record_sha256=residue.source_record_sha256,
                    domain_sha256=domain.domain_sha256,
                    local_axis=local_axis,
                    local_axis_label=local_label,
                    absolute_quadrant=domain.absolute_quadrant,
                    world_sign=world_sign,
                    frame_translation_dyadic=world_translation,
                    leading_cell_coordinate=None,
                    selected_b=selected_b,
                    residue_index=index,
                    source_hypothesis_sha256=None,
                    contributor_order=0)
                add(world_axis, anchor, contributor)

    # 2. P1-B phase-candidate residues named by an accepted source digest.
    accepted_sources = {
        residue.source_record_sha256
        for domain in v2a_result.reading_domains
        for residue in (domain.residue_u, domain.residue_v)}
    domains_by_reading = {}
    for domain in v2a_result.reading_domains:
        identity = domain.reading_identity
        domains_by_reading[(identity.frame, identity.canonical_index,
                            identity.reading_axis)] = domain
    for evidence in sorted(replayed_evidence, key=_evidence_sort_key):
        evidence_sha = _hypothesis_evidence_sha256(evidence)
        for record in evidence.phase_candidates:
            source_sha = _p1b_record_sha256(record)
            if source_sha not in accepted_sources:
                continue
            domain = domains_by_reading.get(
                (record.frame, record.plane_index, record.reading_axis))
            if domain is None:
                continue
            local_axis = record.local_axis
            if local_axis not in _LOCAL_AXES:
                failures[f"{record.frame}:{record.plane_index}"] = \
                    "axis_transform_invalid"
                continue
            # Derive the real tangent label from this reading/domain.  The linear
            # local index is not a world label: for a reading whose normal is the
            # world axis 0, ``u``/``v`` are the labels 1/2, never 0/1.
            residue = domain.residue_u if local_axis == "u" else domain.residue_v
            try:
                expected_label = _reading_tangent_label(
                    record.reading_axis, local_axis)
            except ValueError:
                failures[f"{record.frame}:{record.plane_index}"] = \
                    "axis_transform_invalid"
                continue
            if (residue.local_axis != local_axis
                    or residue.local_axis_label != expected_label
                    or residue.source_record_sha256 != source_sha):
                failures[f"{record.frame}:{record.plane_index}:{local_axis}"] = \
                    "axis_transform_invalid"
                continue
            local_label = residue.local_axis_label
            try:
                world_axis, world_sign = plane_graph.axis_transform(
                    local_label, domain.absolute_quadrant)
            except ValueError:
                failures[f"{record.frame}:{record.plane_index}"] = \
                    "axis_transform_invalid"
                continue
            translation = translations.get(record.frame)
            if translation is None:
                failures[record.frame] = "propagation_unavailable"
                continue
            world_translation = _reduce_dyadic(translation[world_axis])
            selected_b = _reduce_fraction(
                _fraction_from_float(record.blocks_per_period))
            leading = (None if record.leading_cell_coordinate is None
                       else _reduce_dyadic(
                           _fraction_from_float(record.leading_cell_coordinate)))
            for index, raw in enumerate(record.residue_candidates):
                value = _fraction_from_float(raw)
                anchor = _mod1(world_sign * value + translation[world_axis])
                contributor = _make_contributor(
                    source_kind="phase_candidates",
                    source_record_sha256=source_sha,
                    domain_sha256=domain.domain_sha256,
                    local_axis=local_axis,
                    local_axis_label=local_label,
                    absolute_quadrant=domain.absolute_quadrant,
                    world_sign=world_sign,
                    frame_translation_dyadic=world_translation,
                    leading_cell_coordinate=leading,
                    selected_b=selected_b,
                    residue_index=index,
                    source_hypothesis_sha256=evidence_sha,
                    contributor_order=0)
                add(world_axis, anchor, contributor)

    # 3. Accepted C2-R-B edge references, attached to matching candidate anchors.
    for reference in certificate.edge_references:
        identity = reference.match_identity
        factor_sides = (
            (identity.left_frame, identity.left_plane, identity.left_axis, "u", 0),
            (identity.left_frame, identity.left_plane, identity.left_axis, "v", 1),
            (identity.right_frame, identity.right_plane, identity.right_axis, "u", 2),
            (identity.right_frame, identity.right_plane, identity.right_axis, "v", 3),
        )
        for frame, plane_index, axis, local_axis, position in factor_sides:
            domain = None
            for candidate in v2a_result.reading_domains:
                candidate_identity = candidate.reading_identity
                if (candidate_identity.frame == frame
                        and candidate_identity.reading_axis == axis
                        and candidate_identity.canonical_index == plane_index):
                    domain = candidate
                    break
            if domain is None:
                continue
            residue = domain.residue_u if local_axis == "u" else domain.residue_v
            residue_index = reference.residue_indices[position]
            if not 0 <= residue_index < len(residue.candidate_values):
                continue
            if reference.residue_side_sha256s[position] != residue.source_record_sha256:
                continue
            local_label = residue.local_axis_label
            try:
                world_axis, world_sign = plane_graph.axis_transform(
                    local_label, domain.absolute_quadrant)
            except ValueError:
                continue
            translation = translations.get(frame)
            if translation is None:
                continue
            world_translation = _reduce_dyadic(translation[world_axis])
            value = _fraction_from_float(residue.candidate_values[residue_index])
            anchor = _mod1(world_sign * value + translation[world_axis])
            contributor = _make_contributor(
                source_kind="edge_reference",
                source_record_sha256=reference.ref_sha256,
                domain_sha256=domain.domain_sha256,
                local_axis=local_axis,
                local_axis_label=local_label,
                absolute_quadrant=domain.absolute_quadrant,
                world_sign=world_sign,
                frame_translation_dyadic=world_translation,
                leading_cell_coordinate=None,
                selected_b=_reduce_fraction(
                    _fraction_from_float(residue.selected_b)),
                residue_index=residue_index,
                source_hypothesis_sha256=None,
                contributor_order=0)
            add(world_axis, anchor, contributor)

    hypotheses: dict[int, list[V2BAnchorHypothesis]] = {}
    for axis in _WORLD_AXES:
        ordered_values = sorted(contribs[axis])
        records = []
        for value in ordered_values:
            bucket = contribs[axis][value]
            ordered_bucket = []
            for order, item in enumerate(bucket):
                ordered_bucket.append(_make_contributor(
                    source_kind=item.source_kind,
                    source_record_sha256=item.source_record_sha256,
                    domain_sha256=item.domain_sha256,
                    local_axis=item.local_axis,
                    local_axis_label=item.local_axis_label,
                    absolute_quadrant=item.absolute_quadrant,
                    world_sign=item.world_sign,
                    frame_translation_dyadic=item.frame_translation_dyadic,
                    leading_cell_coordinate=item.leading_cell_coordinate,
                    selected_b=item.selected_b,
                    residue_index=item.residue_index,
                    source_hypothesis_sha256=item.source_hypothesis_sha256,
                    contributor_order=order))
            records.append(_make_hypothesis(axis, value, ordered_bucket))
        hypotheses[axis] = records
    # Contract §10 precedence: hypothesis derivation precedes the deterministic
    # plan and preflight, so ``max_hypotheses_per_axis``/``max_hypotheses_total``
    # are enforced by ``_plan_work`` with the exact planned snapshot rather than
    # by an early stop that cannot report exact planned work.
    return hypotheses, failures


class _V2BBudgetStop(Exception):
    """Atomic budget stop carrying the exact work snapshot for its branch.

    A runtime stop carries the truthful runtime ``V2BAnchorWork`` observed at
    the atomic boundary; a planned stop never fabricates a runtime snapshot.
    """

    def __init__(self, cap: str, work: V2BAnchorWork | None = None):
        super().__init__(cap)
        self.cap = cap
        self.work = work


# ---------------------------------------------------------------------------
# Path B determination
# ---------------------------------------------------------------------------

def _observation_inputs(
        world_axis: int, v2a_result, ordered: Mapping[str, tuple],
        replayed_evidence, split_by_reading: Mapping[tuple, Any],
) -> list[dict]:
    """Return the qualifying observation descriptors for one world axis."""
    observations = []
    evidence_by_match = {}
    for evidence in replayed_evidence:
        evidence_by_match.setdefault(
            (evidence.left_frame, evidence.left_plane, evidence.left_axis),
            []).append(evidence)
        evidence_by_match.setdefault(
            (evidence.right_frame, evidence.right_plane, evidence.right_axis),
            []).append(evidence)
    for domain in v2a_result.reading_domains:
        identity = domain.reading_identity
        key = (identity.frame, identity.canonical_index, identity.reading_axis)
        source = split_by_reading.get(key)
        if source is None:
            continue
        for local_axis, residue in (("u", domain.residue_u), ("v", domain.residue_v)):
            local_label = residue.local_axis_label
            try:
                mapped_axis, world_sign = plane_graph.axis_transform(
                    local_label, domain.absolute_quadrant)
            except ValueError:
                continue
            if mapped_axis != world_axis:
                continue
            record = _find_p1b_record(
                replayed_evidence, identity.frame, identity.canonical_index,
                identity.reading_axis, local_axis, residue.source_record_sha256)
            if record is None or record.leading_cell_coordinate is None:
                continue
            observations.append({
                "domain": domain,
                "local_axis": local_axis,
                "residue": residue,
                "record": record,
                "world_sign": world_sign,
                "source": source,
            })
    return observations


def _find_p1b_record(replayed_evidence, frame, plane_index, axis, local_axis,
                     source_sha):
    for evidence in replayed_evidence:
        for record in evidence.phase_candidates:
            if (record.frame == frame and record.plane_index == plane_index
                    and record.reading_axis == axis
                    and record.local_axis == local_axis
                    and _p1b_record_sha256(record) == source_sha):
                return record
    return None


def _compute_band_energies(
        record: plane_residue.PeriodicResidueCandidates,
        seam: lattice_planes.SplitSeamProfileEvidence,
        anchor: Fraction, calibration: V2BAnchorCalibration,
        height: int, width: int,
) -> tuple[Fraction | None, Fraction | None]:
    """Exact inclusive coverage-normalised boundary/interior energies."""
    samples, weights = _decode_seam_profile(seam, height, width)
    period = _fraction_from_float(seam.period_pixels)
    selected_b = _fraction_from_float(record.blocks_per_period)
    leading = _fraction_from_float(record.leading_cell_coordinate)
    tau_band = calibration.tau_band_fraction
    boundary_num = Fraction(0)
    boundary_den = 0
    interior_num = Fraction(0)
    interior_den = 0
    interior_anchor = _mod1(anchor + Fraction(1, 2))
    for index, (sample, weight) in enumerate(zip(samples, weights)):
        x_value = _mod1(leading + selected_b * Fraction(index) / period)
        if _d1(x_value, anchor) <= tau_band:
            boundary_num += _fraction_from_float(sample)
            boundary_den += int(weight)
        if _d1(x_value, interior_anchor) <= tau_band:
            interior_num += _fraction_from_float(sample)
            interior_den += int(weight)
    if boundary_den == 0 or interior_den == 0:
        return None, None
    return boundary_num / boundary_den, interior_num / interior_den


@dataclass
class _SupportAccount:
    """Shared ledger for the exact runtime frame-support insertion counter.

    ``records`` collects every processed observation in canonical axis order;
    ``supports`` is the cardinality of unique ``(world_axis, hypothesis, frame)``
    triples whose frozen test accepted at least one processed observation.
    A triple is counted once, is retained after later rejections, and is never
    rebuilt from the planned upper bound.
    """

    records: list
    supports: int = 0


def _classify_axis(
        world_axis: int, hypotheses: Sequence[V2BAnchorHypothesis],
        observations: Sequence[dict], calibration: V2BAnchorCalibration,
        heights: Mapping[tuple, tuple[int, int]], budget: V2BAnchorBudget,
        account: _SupportAccount,
) -> V2BAnchorAxisResult:
    """Run the frozen absolute boundary/interior test for one world axis.

    Accepted frame supports are counted at insertion against
    ``max_accepted_frame_supports`` and stop atomically on the first ``K > cap``.
    """
    accepted: list[V2BAnchorHypothesis] = []
    rejected: list[V2BAnchorHypothesis] = []
    any_qualifying = False
    max_frames = 0
    for hypothesis in hypotheses:
        all_accept = True
        accepted_frames: set[str] = set()
        for descriptor in observations:
            domain = descriptor["domain"]
            record = descriptor["record"]
            seam = _seam_for(descriptor["source"], descriptor["local_axis"])
            if seam is None:
                continue
            height, width = heights[(domain.reading_identity.frame,
                                     domain.reading_identity.canonical_index,
                                     domain.reading_identity.reading_axis)]
            translation = descriptor["translation"]
            anchor = _mod1(descriptor["world_sign"] * (
                hypothesis.value - translation))
            boundary, interior = _compute_band_energies(
                record, seam, anchor, calibration, height, width)
            zero = boundary is None
            if not zero:
                any_qualifying = True
                held_out = int(seam.coverage["valid"])
                margin = boundary - interior
                accept = (margin >= calibration.tau_margin_fraction
                          and boundary >= calibration.tau_energy_fraction
                          and held_out >= calibration.min_held_out_samples)
                if not accept:
                    all_accept = False
            else:
                held_out = 0
                margin = None
                accept = False
            origin = _reduce_dyadic(_mod1(
                descriptor["world_sign"] * _fraction_from_float(
                    record.leading_cell_coordinate)
                + translation))
            payload = {
                "reading_identity": {
                    "frame": domain.reading_identity.frame,
                    "plane_index": domain.reading_identity.canonical_index,
                    "reading_axis": domain.reading_identity.reading_axis},
                "local_axis": descriptor["local_axis"],
                "world_axis": world_axis,
                "world_sign": descriptor["world_sign"],
                "selected_b": _reduce_fraction(
                    _fraction_from_float(descriptor["residue"].selected_b)),
                "partition_sha256": seam_domain_partition(descriptor["source"]),
                "G_a_sha256": _role_digest(descriptor["source"], descriptor["local_axis"], "G"),
                "D_a_sha256": _role_digest(descriptor["source"], descriptor["local_axis"], "D"),
                "generation_evidence_sha256":
                    descriptor["source"][0].generation_evidence[0].
                    generation_evidence_sha256,
                "seam_evidence_sha256": seam.seam_evidence_sha256,
                "d_profile_sha256": seam.d_profile_sha256,
                "g_reconstruction_sha256":
                    descriptor["g_reconstruction_sha256"],
                "split_residue_binding_sha256":
                    descriptor["split_residue_binding_sha256"],
                "boundary_energy": None if boundary is None
                                   else _reduce_fraction(boundary),
                "interior_energy": None if interior is None
                                   else _reduce_fraction(interior),
                "margin": None if margin is None
                          else _reduce_fraction(margin),
                "zero_coverage": zero,
                "held_out_samples": held_out,
                "hypothesis_sha256": hypothesis.hypothesis_sha256,
                "propagation_origin": origin,
            }
            account.records.append(_seal(
                V2BAnchorObservation, payload, "observation_sha256"))
            if accept and domain.reading_identity.frame not in accepted_frames:
                accepted_frames.add(domain.reading_identity.frame)
                account.supports += 1
                if account.supports > budget.max_accepted_frame_supports:
                    raise _V2BBudgetStop("max_accepted_frame_supports")
        if any_qualifying:
            max_frames = max(max_frames, len(accepted_frames))
        if hypotheses and all_accept and any_qualifying \
                and len(accepted_frames) >= calibration.min_distinct_frames_per_axis:
            accepted.append(hypothesis)
        else:
            rejected.append(hypothesis)

    if not hypotheses:
        axis = _seal(V2BAnchorAxisResult, {
            "world_axis": world_axis, "status": "abstained",
            "alpha_dyadic": None, "reason": "no_hypothesis",
            "accepted_hypotheses": (), "rejected_hypotheses": (),
            "distinct_frame_count": 0}, "axis_sha256")
        return axis
    if not observations or not any_qualifying:
        axis = _seal(V2BAnchorAxisResult, {
            "world_axis": world_axis, "status": "abstained",
            "alpha_dyadic": None, "reason": "no_seam_evidence",
            "accepted_hypotheses": (), "rejected_hypotheses": tuple(rejected),
            "distinct_frame_count": max_frames}, "axis_sha256")
        return axis
    if len(accepted) == 1:
        axis = _seal(V2BAnchorAxisResult, {
            "world_axis": world_axis, "status": "determined",
            "alpha_dyadic": (accepted[0].numerator, accepted[0].denominator),
            "reason": None, "accepted_hypotheses": (accepted[0],),
            "rejected_hypotheses": tuple(rejected),
            "distinct_frame_count": max_frames}, "axis_sha256")
        return axis
    if len(accepted) >= 2:
        reason = "multiple_hypotheses_supported"
    elif len(hypotheses) == 1:
        reason = "evidence_conflict"
    else:
        reason = "no_seam_evidence"
    axis = _seal(V2BAnchorAxisResult, {
        "world_axis": world_axis, "status": "abstained",
        "alpha_dyadic": None, "reason": reason,
        "accepted_hypotheses": tuple(accepted),
        "rejected_hypotheses": tuple(rejected),
        "distinct_frame_count": max_frames}, "axis_sha256")
    return axis


def _seam_for(source: tuple, local_axis: str):
    result = source[0]
    if not isinstance(result, lattice_planes.SplitPhaseProducerResult) \
            or result.status != "complete" or not result.seam_profile_evidence:
        return None
    for seam in result.seam_profile_evidence:
        if seam.local_axis == local_axis:
            return seam
    return None


def _role_digest(source: tuple, local_axis: str, role: str) -> str:
    result = source[0]
    key = {"G": "G", "D": "D"}[role] + "_" + local_axis + "_sha256"
    value = getattr(result, key, None)
    return value if _hex64(value) else "0" * 64


def seam_domain_partition(source: tuple) -> str:
    result = source[0]
    if _hex64(result.partition_sha256):
        return result.partition_sha256
    return "0" * 64


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _refused(reason: str, detail: Any = None,
             work: V2BAnchorWork | None = None) -> V2BAnchorResult:
    return _make_result(status="refused",
                        prediction=None,
                        refusals=(_make_refusal(reason, detail),),
                        budget_stage=None, work=work)


def _budget_result(stage: str, cap: str, limit: int, observed: int,
                   work: V2BAnchorWork | None,
                   planned: V2BAnchorWork | None = None) -> V2BAnchorResult:
    """Atomic budget outcome retaining the exact branch snapshot.

    Planned exhaustion keeps the exact ``planned_work`` as ``work`` and carries
    no runtime snapshot.  Runtime exhaustion keeps the exact runtime snapshot as
    ``work`` and additionally retains the exact planned snapshot in the bounded
    refusal detail, so neither branch fabricates ``observed = planned + 1``.
    """
    detail = {"budget_stage": stage, "cap": cap, "limit": limit,
              "observed": observed}
    if stage == "runtime":
        if work is None or planned is None:
            raise ValueError("v2b_input_invalid")
        detail["planned_work"] = planned.to_record()
        detail["runtime_work"] = work.to_record()
    return _make_result(
        status="budget_exhausted", prediction=None,
        refusals=(_make_refusal("budget_exhausted", detail, cap=cap),),
        budget_stage=stage, work=work)


def _abstained(reason: str, c2rb_result, v2a_result, calibration,
               budget: V2BAnchorBudget,
               split_root: str = "0" * 64,
               rejection: str | None = None) -> V2BAnchorResult:
    work = _make_work({})
    refusals = () if rejection is None else (
        _make_refusal(rejection, {"reason": reason}),)
    axes = tuple(_seal(V2BAnchorAxisResult, {
        "world_axis": axis, "status": "abstained", "alpha_dyadic": None,
        "reason": reason if reason in V2B_ABSTENTION_REASONS
                  else "input_gate_not_unique",
        "accepted_hypotheses": (), "rejected_hypotheses": (),
        "distinct_frame_count": 0}, "axis_sha256") for axis in _WORLD_AXES)
    prediction = V2BAnchorPrediction.build(
        status="abstained", reason=reason,
        alpha_x=None, alpha_y=None, alpha_z=None, axes=axes,
        c2rb_result_sha256=_safe_digest(c2rb_result, "result_sha256"),
        c2rb_input_sha256=_safe_digest(c2rb_result, "input_sha256"),
        vector_result_sha256="0" * 64,
        v2a_result_sha256=_safe_digest(v2a_result, "result_sha256"),
        v2a_input_sha256=_safe_digest(v2a_result, "input_sha256"),
        kernel_result_sha256=_safe_digest(v2a_result, "kernel_result_sha256"),
        canonical_universe_sha256=_safe_digest(
            v2a_result, "canonical_universe_sha256"),
        split_phase_result_sha256=split_root,
        split_phase_result_sha256s=(),
        split_generation_evidence_sha256="0" * 64,
        split_g_phase_reconstruction_sha256="0" * 64,
        split_seam_profile_evidence_sha256="0" * 64,
        split_residue_binding_sha256="0" * 64,
        p1b_hypothesis_phase_shift_evidence_sha256="0" * 64,
        v2a_replay_ok=False, hypotheses=(), observations=(),
        planned_work=work, runtime_work=work, refusals=refusals,
        calibration_sha256=calibration.calibration_sha256,
        budget_sha256=budget.budget_sha256)
    return _make_result(status="abstained", prediction=prediction,
                        refusals=(), budget_stage=None, work=work)


def _safe_digest(value: Any, name: str) -> str:
    candidate = getattr(value, name, None)
    return candidate if _hex64(candidate) else "0" * 64


def _gate_reason(v2a_result) -> str | None:
    if not isinstance(v2a_result, v2a.V2AEndpointResult):
        return "v2b_input_invalid"
    if v2a_result.status != "complete":
        return "input_gate_not_unique"
    if v2a_result.classification == "endpoint_multiple":
        return "endpoint_multiple"
    if v2a_result.classification == "endpoint_none":
        return "endpoint_none"
    if v2a_result.classification != "endpoint_unique":
        return "input_gate_not_unique"
    if (v2a_result.exhaustive is not True or v2a_result.lower_bound_only is not False
            or v2a_result.translation_class_count != 1
            or v2a_result.endpoint_assignment_count != 1
            or len(v2a_result.witnesses) != 1
            or v2a_result.refusals
            or v2a_result.budget_failure_reason is not None):
        return "input_gate_not_unique"
    return None


def _c2rb_gate_reason(c2rb_result, vector_result) -> str | None:
    if (not isinstance(c2rb_result, plane_relative_translation.C2RBResult)
            or not isinstance(vector_result,
                             plane_vector_domain.C2RVectorDomainResult)):
        return "v2b_input_invalid"
    if (c2rb_result.status != "complete"
            or c2rb_result.placement_readiness != "ready"
            or c2rb_result.connected is not True
            or len(c2rb_result.components) != 1):
        return "input_gate_not_unique"
    component = c2rb_result.components[0]
    if (component.outcome != "unique_validated" or component.cycle_rank < 1
            or component.exhaustive is not True or component.failing_chords
            or any(cycle.matching_provenance_count <= 0
                   for cycle in component.cycle_residuals)
            or len(component.translation_classes) != 1):
        return "input_gate_not_unique"
    if (vector_result.status != "complete" or vector_result.refusals
            or not vector_result.domains):
        return "input_gate_not_unique"
    return None


def analyze_phase_anchor(
        c2rb_result,
        vector_result,
        support,
        p1d_result,
        matches: Sequence[plane_graph.PlaneMatch],
        phase_evidence: Sequence[plane_residue.HypothesisPhaseShiftEvidence],
        *,
        readings: Sequence[lattice_planes.PlaneReading],
        signatures: Sequence[lattice_planes.PlaneSignature],
        comparison_tolerance: float,
        closure_tolerance: float,
        v2a_result,
        split_sources: Sequence[tuple],
        budget: V2BAnchorBudget,
        calibration: V2BAnchorCalibration = None,
) -> V2BAnchorResult:
    """Determine (or abstain on) ``alpha`` from exactly one accepted bundle."""
    calibration = V2B_BUILTIN_CALIBRATION if calibration is None else calibration
    if not isinstance(calibration, V2BAnchorCalibration):
        return _refused("calibration_not_builtin")
    if (calibration.calibration_sha256
            != V2B_BUILTIN_CALIBRATION.calibration_sha256):
        return _refused("calibration_not_builtin")
    for tolerance in (comparison_tolerance, closure_tolerance):
        if type(tolerance) not in (int, float) or not math.isfinite(float(tolerance)) \
                or float(tolerance) < 0.0:
            return _refused("v2b_input_invalid",
                            {"field": "tolerance"})
    if not isinstance(budget, V2BAnchorBudget):
        return _refused("v2b_input_invalid", {"field": "budget"})

    # §8 input gate.
    gate = _gate_reason(v2a_result)
    if gate == "v2b_input_invalid":
        return _refused("v2b_input_invalid", {"field": "v2a_result"})
    c2rb_gate = _c2rb_gate_reason(c2rb_result, vector_result)
    if c2rb_gate == "v2b_input_invalid":
        return _refused("v2b_input_invalid", {"field": "c2rb_result"})
    if gate is not None or c2rb_gate is not None:
        return _abstained(gate or c2rb_gate, c2rb_result, v2a_result,
                          calibration, budget)

    # Replay the accepted upstream results from their explicit sources.
    try:
        if not plane_relative_translation.validate_relative_translation_graph(
                c2rb_result, vector_result, support, p1d_result, matches,
                phase_evidence, readings=readings, signatures=signatures,
                comparison_tolerance=comparison_tolerance,
                closure_tolerance=closure_tolerance):
            return _refused("c2rb_replay_mismatch")
        if not v2a.validate_endpoint_provenance_result(
                v2a_result, c2rb_result, vector_result, support, p1d_result,
                matches, phase_evidence, readings=readings,
                signatures=signatures,
                comparison_tolerance=comparison_tolerance,
                closure_tolerance=closure_tolerance):
            return _refused("v2a_replay_mismatch")
    except (AttributeError, TypeError, ValueError, OverflowError,
            FloatingPointError, ZeroDivisionError) as error:
        return _refused("v2b_input_invalid", {"detail": type(error).__name__})

    try:
        ordered = _ordered_readings(readings)
    except ValueError:
        return _refused("v2b_input_invalid", {"field": "readings"})

    # Validate and index the P0 split sources and reconstruct G-only readings.
    try:
        split_by_reading, reconstructions, bindings, split_root, \
            generation_root, seam_root, reconstruction_root, binding_root, \
            sources_work = _validate_split_sources(
                split_sources, ordered, phase_evidence, v2a_result,
                calibration, budget)
    except _V2BFailure as failure:
        return _refused(failure.reason, failure.detail)

    # Replay the whole P1-B unit from the G-only readings.
    try:
        replayed = _replay_p1b(ordered, reconstructions, matches, v2a_result)
        bindings = _build_bindings(split_by_reading, phase_evidence, replayed)
        binding_root = _root_sha256(
            [value.binding_sha256 for value in bindings.values()])
    except _V2BFailure as failure:
        return _refused(failure.reason, failure.detail)

    supplied_root = _root_sha256(sorted(
        _hypothesis_evidence_sha256(value) for value in phase_evidence))
    replayed_root = _root_sha256(sorted(
        _hypothesis_evidence_sha256(value) for value in replayed))
    if supplied_root != replayed_root:
        return _refused("p1b_source_mismatch",
                        {"supplied": supplied_root, "replayed": replayed_root})
    supplied_shas = tuple(sorted(_hypothesis_evidence_sha256(value)
                                 for value in phase_evidence))
    replayed_shas = tuple(sorted(_hypothesis_evidence_sha256(value)
                                 for value in replayed))
    if supplied_shas != replayed_shas:
        return _refused("p1b_source_mismatch")
    phase_candidates_shas = tuple(
        _sha256(record.to_record())
        for evidence in replayed for record in evidence.phase_candidates)

    # Derive H_w with complete ordered contributor provenance.
    try:
        hypotheses, failures = _derive_hypotheses(
            c2rb_result, v2a_result, replayed, ordered, budget)
    except (ValueError, TypeError, OverflowError, FloatingPointError,
            ZeroDivisionError) as error:
        return _refused("v2b_input_invalid", {"detail": type(error).__name__})
    if failures:
        if any(reason == "axis_transform_invalid" for reason in failures.values()):
            return _refused("axis_transform_invalid", dict(sorted(failures.items())))
        return _refused("propagation_unavailable", dict(sorted(failures.items())))
    if not any(hypotheses[axis] for axis in _WORLD_AXES):
        return _abstained("no_hypothesis", c2rb_result, v2a_result,
                          calibration, budget, split_root=split_root)

    # Deterministic observation descriptors: required to record the amended
    # planned upper bound U before classification, and shared with the
    # classifier so both stages use exactly the same qualifying set Q_w.
    observations_by_axis: dict[int, list[dict]] = {
        axis: _observation_inputs(axis, v2a_result, ordered, replayed,
                                  split_by_reading)
        for axis in _WORLD_AXES}

    # Planned preflight before any observation is classified.
    try:
        planned = _plan_work(
            len(v2a_result.reading_domains), hypotheses, split_sources,
            ordered, budget, observations_by_axis)
    except (ValueError, TypeError) as error:
        return _refused("v2b_input_invalid", {"detail": type(error).__name__})
    trigger = _first_over_cap(planned, budget)
    if trigger is not None:
        return _budget_result("planned", trigger,
                              getattr(budget, trigger),
                              getattr(planned, _CAP_TO_WORK[trigger]), planned)

    # Bounded per-hypothesis D-profile classification.
    try:
        axes, observations, supports = _determine(
            c2rb_result, hypotheses, v2a_result, ordered, replayed,
            split_by_reading, reconstructions, bindings, calibration, budget,
            planned, seam_root, observations_by_axis)
    except _V2BBudgetStop as stop:
        runtime = stop.work
        if runtime is None:
            raise
        return _budget_result("runtime", stop.cap,
                              getattr(budget, stop.cap),
                              getattr(runtime, _CAP_TO_WORK[stop.cap]),
                              runtime, planned)
    except _V2BFailure as failure:
        return _refused(failure.reason, failure.detail)

    runtime = _runtime_work(hypotheses, observations, split_by_reading,
                            ordered, planned, supports)
    determined = [axis.status == "determined" for axis in axes]
    status = ("determined" if all(determined) else
              "abstained" if not any(determined) else "partial")
    if status == "determined":
        reason = "determined"
    elif status == "partial":
        reason = "partial"
    else:
        reason = _overall_abstention_reason(axes)
    alphas = tuple(axis.alpha_dyadic if axis.status == "determined" else None
                   for axis in axes)
    prediction = V2BAnchorPrediction.build(
        status=status, reason=reason,
        alpha_x=alphas[0], alpha_y=alphas[1], alpha_z=alphas[2], axes=axes,
        c2rb_result_sha256=c2rb_result.result_sha256,
        c2rb_input_sha256=c2rb_result.input_sha256,
        vector_result_sha256=vector_result.result_sha256,
        v2a_result_sha256=v2a_result.result_sha256,
        v2a_input_sha256=v2a_result.input_sha256,
        kernel_result_sha256=v2a_result.kernel_result_sha256,
        canonical_universe_sha256=v2a_result.canonical_universe_sha256,
        split_phase_result_sha256=split_root,
        split_phase_result_sha256s=sources_work,
        split_generation_evidence_sha256=generation_root,
        split_g_phase_reconstruction_sha256=reconstruction_root,
        split_seam_profile_evidence_sha256=seam_root,
        split_residue_binding_sha256=binding_root,
        p1b_hypothesis_phase_shift_evidence_sha256=replayed_root,
        v2a_replay_ok=True,
        hypotheses=tuple(hypothesis for axis in _WORLD_AXES
                         for hypothesis in hypotheses[axis]),
        observations=tuple(observations),
        planned_work=planned, runtime_work=runtime, refusals=(),
        calibration_sha256=calibration.calibration_sha256,
        budget_sha256=budget.budget_sha256)
    return _make_result(status=status, prediction=prediction, refusals=(),
                        budget_stage=None, work=runtime)


def validate_phase_anchor_replay(
        claimed: V2BAnchorResult,
        c2rb_result,
        vector_result,
        support,
        p1d_result,
        matches: Sequence[plane_graph.PlaneMatch],
        phase_evidence: Sequence[plane_residue.HypothesisPhaseShiftEvidence],
        *,
        readings: Sequence[lattice_planes.PlaneReading],
        signatures: Sequence[lattice_planes.PlaneSignature],
        comparison_tolerance: float,
        closure_tolerance: float,
        v2a_result,
        split_sources: Sequence[tuple],
        budget: V2BAnchorBudget,
        calibration: V2BAnchorCalibration = None,
) -> bool:
    """Source-bound replay: rebuild the record and compare canonical bytes.

    The record constructors can only bind a freshly resealed record to the
    evidence it already retains.  They cannot recover the exact per-reading
    held-out D-coordinate union, nor the exact runtime insertion prefix that a
    stop observed, so a consistently resealed held-out snapshot or a rewritten
    runtime stop is invisible to them.  This validator closes that gap: it
    re-runs the accepted analysis on the original validated split sources, the
    original upstream replay sources and the caller's budget/context, then
    requires the replay's canonical result bytes to equal the claimed record's
    canonical bytes.  Any contradiction -- a changed held-out maximum, a
    rewritten runtime prefix or cap, a branch cap the analysis could not have
    produced -- changes the bytes and is rejected.

    Invalid input keeps the analysis path's invalid-input-before-budget
    precedence: the same call returns the same ``refused`` record before any
    budget cap is evaluated, so a forged ``budget_exhausted`` record over
    invalid sources never validates.  The validator consumes no user-supplied
    geometry, manifest or metadata and therefore moves no advanced-only
    boundary.  A non-``V2BAnchorResult`` claim, or an analysis that raises
    rather than returning a record, is rejected as ``False``.
    """
    if not isinstance(claimed, V2BAnchorResult):
        return False
    try:
        replayed = analyze_phase_anchor(
            c2rb_result, vector_result, support, p1d_result, matches,
            phase_evidence, readings=readings, signatures=signatures,
            comparison_tolerance=comparison_tolerance,
            closure_tolerance=closure_tolerance, v2a_result=v2a_result,
            split_sources=split_sources, budget=budget,
            calibration=calibration)
    except (AttributeError, TypeError, ValueError, OverflowError,
            FloatingPointError, ZeroDivisionError):
        return False
    return replayed.to_json_bytes() == claimed.to_json_bytes()


class _V2BFailure(Exception):
    def __init__(self, reason: str, detail: Any = None):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


def _overall_abstention_reason(axes: Sequence[V2BAnchorAxisResult]) -> str:
    for axis in axes:
        if axis.reason in ("multiple_hypotheses_supported",):
            return "multiple_hypotheses_supported"
    for axis in axes:
        if axis.reason in ("evidence_conflict",):
            return "evidence_conflict"
    for axis in axes:
        if axis.reason in ("no_seam_evidence",):
            return "no_seam_evidence"
    return "input_gate_not_unique"


def _validate_split_sources(
        split_sources, ordered, phase_evidence, v2a_result, calibration, budget,
):
    """Validate P0 records, patches, G reconstruction and the P1-B binding."""
    if not isinstance(split_sources, (list, tuple)):
        raise _V2BFailure("v2b_input_invalid", {"field": "split_sources"})
    # Contract §10 precedence: every source is validated before any budget cap
    # is evaluated, so invalid input refuses even when it also exceeds a cap.
    # ``max_split_reconstructions``/``max_source_records`` are enforced by the
    # deterministic plan in ``_plan_work`` with the exact planned snapshot.
    split_by_reading: dict[tuple, tuple] = {}
    reconstructions: dict[tuple, tuple] = {}
    bindings: dict[tuple, V2BSplitResidueBinding] = {}
    result_shas: list[str] = []
    generation_shas: list[str] = []
    seam_shas: list[str] = []
    reconstruction_shas: list[str] = []
    binding_shas: list[str] = []
    normalized = []
    for index, source in enumerate(split_sources):
        if not isinstance(source, (list, tuple)) or len(source) not in (2, 3):
            raise _V2BFailure("v2b_input_invalid", {"field": "split_sources"})
        result = source[0]
        if not isinstance(result, lattice_planes.SplitPhaseProducerResult) \
                or result.reading_identity is None:
            raise _V2BFailure("v2b_input_invalid", {"field": "split_result"})
        identity = result.reading_identity
        normalized.append(((identity["frame"], identity["plane_index"],
                            identity["reading_axis"]), index, source))
    for key, index, source in sorted(normalized, key=lambda item: item[0]):
        result, patch = source[0], source[1]
        p0_budget = (source[2] if len(source) == 3 else None)
        if result.status != "complete":
            raise _V2BFailure("split_not_reconstructible",
                              {"status": result.status})
        if patch is None:
            raise _V2BFailure("p0_patch_unavailable")
        if p0_budget is None:
            p0_budget = _p0_budget_for(result)
        if not isinstance(p0_budget, lattice_planes.SplitPhaseProducerBudget):
            raise _V2BFailure("v2b_input_invalid", {"field": "p0_budget"})
        identity = result.reading_identity
        if identity is None or result.geometry is None:
            raise _V2BFailure("split_not_reconstructible")
        if key in split_by_reading:
            raise _V2BFailure("split_not_reconstructible", {"duplicate": list(key)})
        try:
            reading = _reading_position(ordered, key[0], key[1], key[2])
        except ValueError:
            raise _V2BFailure("g_phase_reconstruction_refused",
                              {"key": list(key)})
        try:
            array = np.asarray(patch)
            if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
                raise _V2BFailure("split_not_reconstructible")
            array = np.ascontiguousarray(array.astype("<f8", copy=False))
        except _V2BFailure:
            raise
        except (TypeError, ValueError, OverflowError):
            raise _V2BFailure("split_not_reconstructible")
        try:
            ok = lattice_planes.validate_split_phase_evidence(
                result, array, geometry=result.geometry,
                partition_config=lattice_planes.SPLIT_PHASE_PARTITION_CONFIG,
                budget=p0_budget)
        except (AttributeError, TypeError, ValueError):
            ok = False
        if not ok:
            if (int(array.shape[0]) != result.patch_ref.rows
                    or int(array.shape[1]) != result.patch_ref.columns):
                raise _V2BFailure("p0_patch_mismatch")
            raise _V2BFailure("split_replay_mismatch")
        if (int(array.shape[0]) != result.patch_ref.rows
                or int(array.shape[1]) != result.patch_ref.columns
                or result.patch_ref.dtype != "float64"):
            raise _V2BFailure("p0_patch_mismatch")
        generation = result.generation_evidence[0]
        try:
            evidence = _generation_phase_evidence(generation)
        except (ValueError, TypeError) as error:
            raise _V2BFailure("g_phase_reconstruction_refused",
                              {"detail": type(error).__name__})
        try:
            reconstructed = _reconstruct_reading(reading, evidence)
        except (ValueError, TypeError) as error:
            raise _V2BFailure("g_phase_reconstruction_refused",
                              {"detail": type(error).__name__})
        reconstruction = _make_reconstruction(
            result, reading, reconstructed, evidence, generation)
        if not _reading_matches_outside_phase(reading, reconstructed):
            raise _V2BFailure("g_phase_reconstruction_refused")
        split_by_reading[key] = (result, array, generation, reconstruction)
        reconstructions[key] = (reading, reconstructed)
        result_shas.append(result.result_sha256)
        generation_shas.append(generation.generation_evidence_sha256)
        for seam in result.seam_profile_evidence:
            seam_shas.append(seam.seam_evidence_sha256)
        reconstruction_shas.append(reconstruction.reconstruction_sha256)
    for key, (result, array, generation, reconstruction) in split_by_reading.items():
        pass
    return (split_by_reading, reconstructions, bindings,
            _root_sha256(result_shas), _root_sha256(generation_shas),
            _root_sha256(seam_shas), _root_sha256(reconstruction_shas),
            _root_sha256(binding_shas), tuple(result_shas))


def _p0_budget_for(result: lattice_planes.SplitPhaseProducerResult):
    """Rebuild the P0 budget caps that the accepted record must fit."""
    rows = result.patch_ref.rows
    columns = result.patch_ref.columns
    samples = rows * columns
    inline = samples if result.patch_ref.mode == "inline" else max(1, samples - 1)
    return lattice_planes._make_budget({
        "max_lanes_per_axis": max(1, rows, columns),
        "max_patch_samples": samples,
        "max_inline_patch_samples": inline,
        "max_g_samples_per_axis": samples,
        "max_d_samples_per_axis": samples,
        "max_guard_samples_per_axis": max(1, samples),
        "max_profile_terms": rows + columns,
        "max_d_profile_terms": 3 * (rows + columns),
        "max_payload_bytes": (2 * len(
            lattice_planes.SPLIT_SEAM_PROFILE_PAYLOAD_VERSION) + 48
            + 24 * (rows + columns)),
        "max_source_records": len(result.lane_sets) + 3,
        "max_replay_checks": len(result.lane_sets) + 4,
    })


def _reconstruct_reading(reading, evidence):
    return _dataclass_replace(
        reading, phase_evidence=evidence,
        phase_pixels=_phase_pixels_from_evidence(evidence))


def _make_reconstruction(result, reading, reconstructed, evidence, generation):
    identity = result.reading_identity
    payload = {
        "reading_identity": dict(identity),
        "phase_uv_pixels": tuple(evidence.phase_uv_pixels),
        "response_magnitude_uv": tuple(evidence.response_magnitude_uv),
        "profile_mass_uv": tuple(evidence.profile_mass_uv),
        "concentration_uv": tuple(evidence.concentration_uv),
        "unavailable_reason_uv": tuple(evidence.unavailable_reason_uv),
        "period_pixels": evidence.period_pixels,
        "patch_centre_uv": tuple(evidence.patch_centre_uv),
        "extent": evidence.extent,
        "samples": evidence.samples,
        "source_region_id": evidence.source_region_id,
        "partition_sha256": generation.partition_sha256,
        "G_u_sha256": generation.G_u_sha256,
        "G_v_sha256": generation.G_v_sha256,
        "g_profile_sha256": generation.g_profile_sha256,
        "accepted_reading_sha256": _reading_sha256(reading),
        "reconstructed_reading_sha256": _reading_sha256(reconstructed),
    }
    return _seal(V2BGPhaseReconstruction, payload, "reconstruction_sha256")


def _reading_matches_outside_phase(reading, reconstructed) -> bool:
    first = _reading_record(reading)
    second = _reading_record(reconstructed)
    first["phase_evidence"] = None
    second["phase_evidence"] = None
    first["phase_pixels"] = None
    second["phase_pixels"] = None
    return V2B_CANONICAL_JSON(first) == V2B_CANONICAL_JSON(second)


def _build_bindings(split_by_reading, supplied_evidence, replayed_evidence):
    """Bind each P0 result to the whole supplied/replayed P1-B evidence."""
    supplied_by_key = {}
    for evidence in supplied_evidence:
        for key in _evidence_keys(evidence):
            supplied_by_key.setdefault(key, []).append(evidence)
    replayed_by_key = {}
    for evidence in replayed_evidence:
        for key in _evidence_keys(evidence):
            replayed_by_key.setdefault(key, []).append(evidence)
    bindings = {}
    for key, (result, _array, generation, reconstruction) in split_by_reading.items():
        supplied = sorted(supplied_by_key.get(key, []), key=_evidence_sort_key)
        replayed = sorted(replayed_by_key.get(key, []), key=_evidence_sort_key)
        if not supplied or not replayed:
            raise _V2BFailure("p1b_source_mismatch", {"key": list(key)})
        supplied_shas = tuple(_hypothesis_evidence_sha256(value)
                              for value in supplied)
        replayed_shas = tuple(_hypothesis_evidence_sha256(value)
                              for value in replayed)
        if supplied_shas != replayed_shas:
            raise _V2BFailure("p1b_source_mismatch", {"key": list(key)})
        candidates = tuple(
            _sha256(record.to_record())
            for evidence in replayed for record in evidence.phase_candidates)
        bindings[key] = _seal(V2BSplitResidueBinding, {
            "split_phase_result_sha256": result.result_sha256,
            "generation_evidence_sha256": generation.generation_evidence_sha256,
            "g_reconstruction_sha256": reconstruction.reconstruction_sha256,
            "supplied_evidence_root_sha256": _root_sha256(supplied_shas),
            "replayed_evidence_root_sha256": _root_sha256(replayed_shas),
            "phase_candidates_sha256": candidates,
            "supplied_evidence_sha256s": supplied_shas,
            "replayed_evidence_sha256s": replayed_shas,
        }, "binding_sha256")
    return bindings


def _evidence_keys(evidence):
    return (
        (evidence.left_frame, evidence.left_plane, evidence.left_axis),
        (evidence.right_frame, evidence.right_plane, evidence.right_axis),
    )


def _evidence_sort_key(evidence):
    return (evidence.left_frame, evidence.left_plane, evidence.left_axis,
            evidence.right_frame, evidence.right_plane, evidence.right_axis)


def _replay_p1b(ordered, reconstructions, matches, v2a_result):
    """Replay the whole P1-B unit from the reconstructed G-only readings."""
    by_index = {}
    for frame, values in ordered.items():
        for index, reading in enumerate(values):
            key = (frame, index, reading.axis)
            replacement = reconstructions.get(key)
            by_index[(frame, index)] = (reading if replacement is None
                                        else replacement[1])
    replayed = []
    for match in matches:
        replayed.append(plane_residue.hypothesis_phase_shift_evidence(
            match, by_index))
    return tuple(replayed)


def _canonical_index(ordered, reading):
    values = ordered.get(reading.frame, ())
    for index, candidate in enumerate(values):
        if lattice_planes.reading_order_key(candidate) \
                == lattice_planes.reading_order_key(reading):
            return index
    raise _V2BFailure("g_phase_reconstruction_refused")


def _d_coordinate_union_size(result: lattice_planes.SplitPhaseProducerResult) -> int:
    """Cardinality of one reading's distinct ordered held-out D coordinates.

    The union is over the accepted ``D`` lane sets of both local axes, using the
    same ordered ``(row, column)`` coordinate sets that produce the frozen
    ``D_a_sha256`` coordinate digests: rows of the ``u`` D block across every
    column, columns of the ``v`` D block across every row.  Repeated and
    overlapping coordinates count once.
    """
    rows = int(result.patch_ref.rows)
    columns = int(result.patch_ref.columns)
    coordinates: set[tuple[int, int]] = set()
    for lane in result.lane_sets:
        if lane.role != "D":
            continue
        if lane.local_axis == "u":
            for row in range(int(lane.lane_index_start), int(lane.lane_index_stop)):
                for column in range(columns):
                    coordinates.add((row, column))
        else:
            for column in range(int(lane.lane_index_start),
                                int(lane.lane_index_stop)):
                for row in range(rows):
                    coordinates.add((row, column))
    return len(coordinates)


def _processed_reading_keys(observations) -> set:
    keys = set()
    for observation in observations:
        identity = observation.reading_identity
        keys.add((identity["frame"], identity["plane_index"],
                  identity["reading_axis"]))
    return keys


def _max_held_out_union(split_by_reading, keys) -> int:
    result = 0
    for key in keys:
        source = split_by_reading.get(key)
        if source is None:
            continue
        result = max(result, _d_coordinate_union_size(source[0]))
    return result


def _plan_work(reading_count, hypotheses, split_sources, ordered, budget,
               observations_by_axis):
    """Exact deterministic plan, including the amended planned upper bound U.

    ``accepted_frame_supports`` records ``U = sum_w |H_w| * |{frame(o): o in Q_w}|``
    without being compared to ``max_accepted_frame_supports``; the fifteen
    existing static counters remain preflight-enforced.
    """
    total_hypotheses = sum(len(value) for value in hypotheses.values())
    max_per_axis = max((len(value) for value in hypotheses.values()), default=0)
    contributor_per = max(
        (value.contributor_count for axis in _WORLD_AXES
         for value in hypotheses[axis]), default=0)
    contributor_total = sum(
        value.contributor_count for axis in _WORLD_AXES
        for value in hypotheses[axis])
    frames = len(ordered)
    readings = sum(len(values) for values in ordered.values())
    source_records = len(split_sources)
    d_terms = 0
    payload_bytes = 0
    pixel_samples = 0
    for source in split_sources:
        result = source[0]
        rows, columns = result.patch_ref.rows, result.patch_ref.columns
        d_terms += 3 * (rows + columns)
        payload_bytes += (2 * len(lattice_planes.SPLIT_SEAM_PROFILE_PAYLOAD_VERSION)
                          + 48 + 24 * (rows + columns))
        pixel_samples += rows * columns
    observations = 0
    accepted_frame_supports = 0
    planned_keys = set()
    for axis in _WORLD_AXES:
        observations += len(hypotheses[axis]) * _axis_observation_count(
            axis, reading_count)
        frames_for_axis = set()
        for descriptor in observations_by_axis[axis]:
            identity = descriptor["domain"].reading_identity
            frames_for_axis.add(identity.frame)
            planned_keys.add((identity.frame, identity.canonical_index,
                              identity.reading_axis))
        accepted_frame_supports += len(hypotheses[axis]) * len(frames_for_axis)
    held_out = _max_held_out_union(
        {_source_key(source[0]): source for source in split_sources},
        planned_keys)
    return _make_work({
        "readings": readings,
        "observations": observations,
        "frames": frames,
        "hypotheses_per_axis": max_per_axis,
        "hypotheses_total": total_hypotheses,
        "contributor_refs_per_hypothesis": contributor_per,
        "contributor_refs_total": contributor_total,
        "source_records": source_records,
        "boundary_interior_terms": observations * 2,
        "d_profile_terms": d_terms,
        "payload_bytes": payload_bytes,
        "replay_crosslinks": source_records * 2,
        "pixel_samples": pixel_samples,
        "held_out_samples_per_reading": held_out,
        "split_reconstructions": len(split_sources),
        "accepted_frame_supports": accepted_frame_supports,
    })


def _source_key(result) -> tuple:
    identity = result.reading_identity
    return (identity["frame"], identity["plane_index"], identity["reading_axis"])


def _axis_observation_count(world_axis, reading_count):
    # Conservative upper bound; the actual count is data-dependent.
    return max(1, reading_count)


def _runtime_work(hypotheses, observations, split_by_reading, ordered, planned,
                  accepted_frame_supports):
    """Exact runtime snapshot over the observations actually processed."""
    contributor_per = max(
        (value.contributor_count for axis in _WORLD_AXES
         for value in hypotheses[axis]), default=0)
    contributor_total = sum(
        value.contributor_count for axis in _WORLD_AXES
        for value in hypotheses[axis])
    held_out = _max_held_out_union(
        split_by_reading, _processed_reading_keys(observations))
    return _make_work({
        "readings": sum(len(values) for values in ordered.values()),
        "observations": len(observations),
        "frames": len(ordered),
        "hypotheses_per_axis": max(
            (len(hypotheses[axis]) for axis in _WORLD_AXES), default=0),
        "hypotheses_total": sum(
            len(hypotheses[axis]) for axis in _WORLD_AXES),
        "contributor_refs_per_hypothesis": contributor_per,
        "contributor_refs_total": contributor_total,
        "source_records": len(split_by_reading),
        "boundary_interior_terms": len(observations) * 2,
        "d_profile_terms": planned.d_profile_terms,
        "payload_bytes": planned.payload_bytes,
        "replay_crosslinks": planned.replay_crosslinks,
        "pixel_samples": planned.pixel_samples,
        "held_out_samples_per_reading": held_out,
        "split_reconstructions": len(split_by_reading),
        "accepted_frame_supports": accepted_frame_supports,
    })


def _determine(c2rb_result, hypotheses, v2a_result, ordered, replayed,
               split_by_reading, reconstructions, bindings, calibration,
               budget, planned, seam_root, observations_by_axis):
    heights = {}
    for key, (result, _patch, _generation, _reconstruction) in split_by_reading.items():
        heights[key] = (result.patch_ref.rows, result.patch_ref.columns)
    _component, certificate = _certificate_of(c2rb_result)
    translations_by_frame = _translations_by_frame(certificate)
    axes = []
    account = _SupportAccount(records=[])

    def snapshot() -> V2BAnchorWork:
        return _runtime_work(hypotheses, tuple(account.records),
                             split_by_reading, ordered, planned,
                             account.supports)

    for axis in _WORLD_AXES:
        descriptors = observations_by_axis[axis]
        for descriptor in descriptors:
            identity = descriptor["domain"].reading_identity
            key = (identity.frame, identity.canonical_index,
                   identity.reading_axis)
            descriptor["translation"] = translations_by_frame[identity.frame][axis]
            descriptor["g_reconstruction_sha256"] = \
                split_by_reading[key][3].reconstruction_sha256
            descriptor["split_residue_binding_sha256"] = \
                bindings[key].binding_sha256
        try:
            axis_result = _classify_axis(
                axis, hypotheses[axis], descriptors, calibration, heights,
                budget, account)
        except _V2BBudgetStop as stop:
            raise _V2BBudgetStop(stop.cap, snapshot())
        axes.append(axis_result)
        if len(account.records) > budget.max_observations:
            raise _V2BBudgetStop("max_observations", snapshot())
        if len(account.records) * 2 > budget.max_boundary_interior_terms:
            raise _V2BBudgetStop("max_boundary_interior_terms", snapshot())
    return tuple(axes), tuple(account.records), account.supports


# ---------------------------------------------------------------------------
# Path A: consistency-only diagnostic (never determines)
# ---------------------------------------------------------------------------

def diagnose_full_patch_consistency(
        readings: Sequence[lattice_planes.PlaneReading],
        hypotheses_by_axis: Mapping[int, Sequence[V2BAnchorHypothesis]],
        *,
        calibration: V2BAnchorCalibration = None,
) -> V2BConsistencyDiagnostic:
    """Path A: retained full-patch evidence consistency with ``H_w``.

    Non-promotable.  Never emits ``determined`` and never sets ``alpha``.
    """
    calibration = V2B_BUILTIN_CALIBRATION if calibration is None else calibration
    evidence_root = _root_sha256(sorted(
        _sha256(_phase_evidence_record(reading.phase_evidence))
        for reading in readings if reading.phase_evidence is not None))
    periods = []
    axes = []
    for axis in _WORLD_AXES:
        candidates = list(hypotheses_by_axis.get(axis, ()))
        if not candidates:
            axes.append({"world_axis": axis, "status": "insufficient_data",
                         "consistent": 0, "inconsistent": 0, "unavailable": 0})
            continue
        consistent = inconsistent = unavailable = 0
        for reading in readings:
            evidence = reading.phase_evidence
            if evidence is None:
                unavailable += 1
                continue
            periods.append(float(evidence.period_pixels))
            matched = False
            for local_index, local_axis in enumerate(_LOCAL_AXES):
                local_label = 0 if local_axis == "u" else 1
                try:
                    _world, sign = plane_graph.axis_transform(local_label, 0)
                except ValueError:
                    continue
                if _world != axis:
                    continue
                phase = evidence.phase_uv_pixels[local_index]
                if phase is None or evidence.period_pixels <= 0.0:
                    unavailable += 1
                    continue
                implied = _fraction_from_float(phase) / \
                    _fraction_from_float(evidence.period_pixels)
                for hypothesis in candidates:
                    if _d1(implied, hypothesis.value) <= calibration.tau_band_fraction:
                        matched = True
                        break
            if matched:
                consistent += 1
            else:
                inconsistent += 1
        if consistent and not inconsistent:
            status = "consistent"
        elif inconsistent and not consistent:
            status = "inconsistent"
        elif consistent or inconsistent:
            status = "consistent"
        else:
            status = "unavailable"
        axes.append({"world_axis": axis, "status": status,
                     "consistent": consistent, "inconsistent": inconsistent,
                     "unavailable": unavailable})
    statuses = {value["status"] for value in axes}
    if statuses <= {"insufficient_data"}:
        status = "insufficient_data"
    elif statuses <= {"unavailable"}:
        status = "unavailable"
    elif statuses == {"inconsistent"}:
        status = "inconsistent"
    elif "consistent" in statuses:
        status = "consistent"
    elif "inconsistent" in statuses:
        status = "inconsistent"
    else:
        status = "unavailable"
    return _seal(V2BConsistencyDiagnostic, {
        "status": status, "axes": tuple(axes),
        "period_pixels_binding": tuple(periods),
        "full_patch_evidence_root_sha256": evidence_root,
    }, "diagnostic_sha256")


def _phase_evidence_record(evidence) -> dict:
    return {
        "phase_uv_pixels": list(evidence.phase_uv_pixels),
        "period_pixels": float(evidence.period_pixels),
        "patch_centre_uv": list(evidence.patch_centre_uv),
        "extent": float(evidence.extent),
        "samples": int(evidence.samples),
        "source_region_id": evidence.source_region_id,
    }
