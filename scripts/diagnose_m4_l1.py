"""Label-blind diagnostics for frozen image-only plane/lattice runs.

Inputs are one strict runner freeze and its explicit ordered RGB paths. The
script reads no capture, manifest, GT, reference, or adjacent metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from mc_photogrammetry.reconstruction import frozen_image_runner
from mc_photogrammetry.reconstruction.calibration import axis_directions
from mc_photogrammetry.vision.gui_mask import create_gui_mask

EXPECTED_INVENTORY = (
    ("view-canal-town-architectural-component-bbox-00.png", 1280, 720,
     "48ba0ff58c77b588acd5aa06a2c630fc2f5103c6d0d67b45b4bc21b81ecf533c"),
    ("view-canal-town-architectural-component-bbox-01.png", 1600, 900,
     "8750bb197329eb50144896264db40eef70a6c44cc3dce60eac3a4f1e9aa79696"),
    ("view-canal-town-architectural-component-bbox-02.png", 1920, 1080,
     "2d017c8f804999446f54dbc742fb87d7748af76ad50249c0e2853754c55b6a95"),
    ("view-canal-town-architectural-component-bbox-03.png", 1920, 1080,
     "e71bfda3ece7a0b74d2da2fd62834645fcbcf662fb4b5889fae95fa26c9a09e9"),
    ("view-canal-town-architectural-component-bbox-04.png", 1280, 720,
     "8797d70e0e09475d296c3cf18e5892f6b22d41750ed18d6c3364da80ba60a81f"),
    ("view-canal-town-architectural-component-bbox-05.png", 1600, 900,
     "6f844d5e3600e853bf3045fc25b55a271c0b665226053602b6d30c945f0f016b"),
    ("view-canal-town-architectural-component-bbox-06.png", 1600, 900,
     "53c38526d378a0597ab6e00fb5e01a6c7aa0e2dc11619c1b7eaa3cfc0ccf8923"),
    ("view-canal-town-architectural-component-bbox-07.png", 1280, 720,
     "fdd8a3709479795b204f67153b10f44b49e85f2bc0050a46d9fdd0461f025556"),
)

# Frozen legacy measurement constants and arithmetic. This diagnostic intentionally
# does not import the edited lattice/lattice_planes implementation: R2 parity is
# computed from these self-contained legacy operations, while the production
# repair is measured separately by R3.
GRAZING_INCIDENCE = 0.05
MAX_RECTIFIED_SIDE = 2048
OVERSAMPLE = 2.0
MINIMUM_PERIOD = 4.0
MINIMUM_CONTRAST = 1e-6
HARMONIC_SHARE = 0.75


@dataclass(frozen=True)
class _LegacyEstimate:
    period_pixels: float
    strength: float


def _legacy_sampling_for(extent: float, focal: float) -> int:
    return int(max(64, min(8192, round(2.0 * extent * focal * OVERSAMPLE))))


def _legacy_basis(axis: int, axes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    others = [index for index in range(3) if index != axis]
    return axes[axis], axes[others[0]], axes[others[1]]


def _legacy_rectify(
    image: np.ndarray,
    axis: int,
    axes: np.ndarray,
    focal: float,
    *,
    extent: float,
    samples: int,
    principal: tuple[float, float],
    centre: tuple[float, float],
    side: float,
) -> np.ndarray | None:
    height, width = image.shape[:2]
    centre_x, centre_y = principal
    normal, first, second = _legacy_basis(axis, axes)
    grid_u, grid_v = np.meshgrid(
        np.linspace(centre[0] - extent, centre[0] + extent, samples),
        np.linspace(centre[1] - extent, centre[1] + extent, samples))
    points = (side * normal[None, None, :]
              + grid_u[..., None] * first[None, None, :]
              + grid_v[..., None] * second[None, None, :])
    depth = points[..., 2]
    valid = depth > 1e-6
    if not valid.any():
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        xs = centre_x + points[..., 0] * focal / depth
        ys = centre_y - points[..., 1] * focal / depth
    xs = np.where(valid, xs, -1.0).astype(np.float32)
    ys = np.where(valid, ys, -1.0).astype(np.float32)
    return cv2.remap(image, xs, ys, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _legacy_profile(patch: np.ndarray, along: int,
                    weights: np.ndarray | None = None) -> np.ndarray:
    grey = patch if patch.ndim == 2 else patch.mean(axis=2)
    gradient = np.abs(np.diff(grey, axis=1 - along))
    if weights is None:
        return gradient.mean(axis=along)
    mask = weights if weights.ndim == 2 else weights.mean(axis=2)
    mask = mask[:, :-1] if along == 0 else mask[:-1, :]
    total = mask.sum(axis=along)
    covered = (gradient * mask).sum(axis=along)
    return np.divide(covered, total, out=np.zeros_like(covered), where=total > 0)


def _legacy_estimate_period(
    patch: np.ndarray,
    along: int,
    weights: np.ndarray | None = None,
) -> _LegacyEstimate | None:
    signal = _legacy_profile(patch, along, weights)
    signal = signal - signal.mean()
    if signal.size < 8 or float(signal.var()) < MINIMUM_CONTRAST:
        return None
    correlation = np.correlate(signal, signal, mode="full")[signal.size - 1:]
    if correlation[0] <= 0:
        return None
    correlation = correlation / correlation[0]
    low = max(int(np.floor(MINIMUM_PERIOD)), 1)
    high = min(int(np.ceil(signal.size / 3.0)), correlation.size - 1)
    if high <= low:
        return None
    interior = ((correlation[1:-1] >= correlation[:-2])
                & (correlation[1:-1] >= correlation[2:]))
    peaks = np.flatnonzero(interior) + 1
    peaks = peaks[(peaks >= low) & (peaks < high)]
    if peaks.size == 0:
        return None
    tallest = float(correlation[peaks].max())
    if tallest <= 0.0:
        return None
    candidates = peaks[correlation[peaks] >= HARMONIC_SHARE * tallest]
    peak = int(candidates[0]) if candidates.size else int(peaks[np.argmax(correlation[peaks])])
    strength = float(correlation[peak])
    if 1 <= peak < correlation.size - 1:
        before, middle, after = correlation[peak - 1], correlation[peak], correlation[peak + 1]
        denominator = before - 2.0 * middle + after
        if abs(denominator) > 1e-12:
            peak = peak + 0.5 * (before - after) / denominator
    period = float(peak)
    if period < MINIMUM_PERIOD - 1.0:
        return None
    return _LegacyEstimate(period_pixels=period, strength=strength)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_pixels(path: Path, record: dict[str, Any]) -> np.ndarray:
    payload = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not decode explicit image input {path.name}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    if (path.name != record["basename"] or width != record["width"]
            or height != record["height"] or _sha256(payload) != record["sha256"]):
        raise ValueError(f"image identity differs from freeze for {path.name}")
    return image


def _mask_from_ref(sidecar: bytes, descriptor: dict[str, Any]) -> np.ndarray:
    start = int(descriptor["offset"])
    end = start + int(descriptor["length"])
    height, width = (int(value) for value in descriptor["shape"])
    raw = sidecar[start:end]
    return np.unpackbits(np.frombuffer(raw, np.uint8), bitorder="little",
                         count=height * width).reshape(height, width).astype(bool)


def _window_stats(mask: np.ndarray, axis: int, axes: np.ndarray, focal: float,
                  principal: tuple[float, float]) -> dict[str, Any]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return {"pixels": 0, "stable_pixels": 0, "grazing_pixels": 0,
                "opposite_side_pixels": 0, "legacy_extent": None,
                "stable_extent": None, "legacy_side": None}
    height, width = mask.shape
    raw_rays = np.stack(((xs - principal[0]) / focal,
                         -(ys - principal[1]) / focal,
                         np.ones(len(xs), dtype=float)), axis=1)
    ray_lengths = np.linalg.norm(raw_rays, axis=1)
    rays = raw_rays / ray_lengths[:, None]
    normal, first, second = _legacy_basis(axis, axes)
    normal = normal / np.linalg.norm(normal)
    dot = rays @ normal
    raw_dot = raw_rays @ normal
    forward = int(np.count_nonzero(raw_dot > 0.0))
    backward = int(np.count_nonzero(raw_dot < 0.0))
    side = 1.0 if forward >= backward else -1.0
    incidence = dot * side
    legacy = raw_dot * side > 1e-6
    stable = incidence > GRAZING_INCIDENCE

    def extent(valid: np.ndarray, denominator: np.ndarray) -> float | None:
        if not np.any(valid):
            return None
        points = raw_rays[valid] / denominator[valid, None]
        u, v = points @ first, points @ second
        half_span = max(float(np.ptp(u)), float(np.ptp(v))) / 2.0 * 1.05
        return half_span if np.isfinite(half_span) else None

    same_side = incidence > 0.0
    return {
        "pixels": int(len(xs)),
        "dominant_side_positive": forward,
        "dominant_side_negative": backward,
        "stable_pixels": int(np.count_nonzero(stable)),
        "grazing_pixels": int(np.count_nonzero(same_side & (incidence <= GRAZING_INCIDENCE))),
        "opposite_side_pixels": int(np.count_nonzero(~same_side)),
        "stable_fraction": float(np.count_nonzero(stable) / len(xs)),
        "grazing_fraction": float(np.count_nonzero(same_side & (incidence <= GRAZING_INCIDENCE)) / len(xs)),
        "legacy_extent": extent(legacy, raw_dot * side),
        "stable_extent": extent(stable, raw_dot * side),
        "legacy_side": side,
    }


def _selected_period(estimates: list[tuple[int, _LegacyEstimate | None]]) -> tuple[int, _LegacyEstimate] | None:
    found = [(direction, estimate) for direction, estimate in estimates if estimate is not None]
    if not found:
        return None
    if len(found) == 1:
        return found[0]
    ordered = sorted(found, key=lambda item: item[1].period_pixels)
    short, long = ordered
    factor = long[1].period_pixels / short[1].period_pixels
    nearest = round(factor)
    if nearest >= 2 and abs(factor - nearest) <= 0.12:
        return short
    return max(found, key=lambda item: item[1].strength)


def _weighted_profile(patch: np.ndarray, along: int, weights: np.ndarray) -> np.ndarray:
    grey = patch if patch.ndim == 2 else patch.mean(axis=2)
    gradient = np.abs(np.diff(grey, axis=1 - along))
    mask = weights if weights.ndim == 2 else weights.mean(axis=2)
    mask = mask[:, :-1] if along == 0 else mask[:-1, :]
    total = mask.sum(axis=along)
    covered = (gradient * mask).sum(axis=along)
    profile = np.divide(covered, total, out=np.zeros_like(covered), where=total > 0)
    return profile - profile.mean()


def _correlation_support(profile: np.ndarray, period: float, multiple: int) -> float:
    if profile.size < 8 or float(profile.var()) < MINIMUM_CONTRAST:
        return 0.0
    correlation = np.correlate(profile, profile, mode="full")[profile.size - 1:]
    if correlation[0] <= 0.0:
        return 0.0
    index = int(round(period * multiple))
    if index <= 0 or index >= correlation.size:
        return 0.0
    return float(max(0.0, correlation[index] / correlation[0]))


def _reading_diagnostics(image: np.ndarray, usable: np.ndarray, reading: dict[str, Any],
                         proposal_mask: np.ndarray) -> dict[str, Any]:
    rect = reading["rectification"]
    axes = np.asarray(rect["axis_matrix"], dtype=float)
    focal = float(rect["focal"])
    centre = tuple(float(value) for value in rect["centre"])
    principal = tuple(float(value) for value in rect["principal_point"])
    axis = int(reading["axis"])
    side = float(reading["side"])
    extent = float(reading["extent"])
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    grey[~usable] = 0
    initial_request = _legacy_sampling_for(extent, focal)
    samples = min(max(initial_request, 64), MAX_RECTIFIED_SIDE)
    actual_measured_samples: int | None = None
    patch = weights = None
    estimates: list[tuple[int, _LegacyEstimate | None]] = []
    winner: tuple[int, _LegacyEstimate] | None = None
    rounds = []
    for round_index in range(3):
        measured = samples
        patch = _legacy_rectify(
            grey, axis, axes, focal, extent=extent, samples=measured,
            principal=principal, centre=centre, side=side)
        weights = _legacy_rectify(
            proposal_mask.astype(np.float32), axis, axes, focal, extent=extent,
            samples=measured, principal=principal, centre=centre, side=side)
        actual_measured_samples = measured
        if patch is None or weights is None or float(weights.sum()) <= 0.0:
            rounds.append({"round": round_index, "samples": measured,
                           "periods": [None, None], "reason": "rectification_unavailable"})
            return {"period_recomputed": False, "reason": "rectification_unavailable",
                    "serialized_samples": int(reading["samples"]),
                    "actual_measured_samples": measured,
                    "legacy_returned_samples": samples, "round_trace": rounds}
        estimates = [(direction, _legacy_estimate_period(
            patch, along=direction, weights=weights)) for direction in (0, 1)]
        winner = _selected_period(estimates)
        rounds.append({
            "round": round_index,
            "samples": measured,
            "direction_periods": [None if item is None else float(item.period_pixels)
                                  for _, item in estimates],
            "selected_direction": None if winner is None else winner[0],
            "selected_period_pixels": None if winner is None else float(winner[1].period_pixels),
        })
        if winner is None or winner[1].period_pixels >= 10.0:
            break
        wanted = int(round(measured * 10.0 / max(winner[1].period_pixels, 1e-6)))
        wanted = min(max(wanted, 64), MAX_RECTIFIED_SIDE)
        if wanted <= measured:
            break
        # This models the legacy loop's next requested size. If this occurs on
        # round three, it is not a measured resolution and must be kept distinct.
        samples = wanted
    if winner is None or patch is None or weights is None:
        return {"period_recomputed": False, "reason": "no_period",
                "serialized_samples": int(reading["samples"]),
                "actual_measured_samples": actual_measured_samples,
                "legacy_returned_samples": samples, "round_trace": rounds}
    direction, estimate = winner
    horizontal_unweighted = _correlation_support(
        _weighted_profile(patch, 0, np.ones_like(weights)), estimate.period_pixels, 2)
    winning_unweighted = _correlation_support(
        _weighted_profile(patch, direction, np.ones_like(weights)), estimate.period_pixels, 2)
    winning_weighted = _correlation_support(
        _weighted_profile(patch, direction, weights), estimate.period_pixels, 2)
    return {
        "period_recomputed": True,
        "stored_period_pixels": float(reading["period_pixels"]),
        "recomputed_period_pixels": float(estimate.period_pixels),
        "period_relative_error": float(abs(estimate.period_pixels - float(reading["period_pixels"]))
                                        / float(reading["period_pixels"])),
        "initial_sampling_request": initial_request,
        "serialized_samples": int(reading["samples"]),
        "actual_measured_samples": actual_measured_samples,
        "legacy_returned_samples": samples,
        "round_trace": rounds,
        "winning_profile_direction": direction,
        "direction_periods": [None if item is None else float(item.period_pixels)
                              for _, item in estimates],
        "direction_strengths": [None if item is None else float(item.strength)
                                for _, item in estimates],
        "winning_unweighted_support_b2": winning_unweighted,
        "winning_weighted_support_b2": winning_weighted,
        "legacy_horizontal_unweighted_support_b2": horizontal_unweighted,
        "support_difference_weighted_minus_legacy": winning_weighted - horizontal_unweighted,
        "weights_coverage_fraction": float(np.count_nonzero(weights > 0.5) / weights.size),
    }


def diagnose(freeze_path: Path, image_paths: list[Path]) -> dict[str, Any]:
    document = frozen_image_runner.load_and_verify_frozen_image(freeze_path)
    actual_inventory = tuple((item["basename"], int(item["width"]), int(item["height"]),
                              item["sha256"]) for item in document["inputs"])
    if actual_inventory != EXPECTED_INVENTORY:
        raise ValueError("freeze inventory differs from preregistered ordered images")
    if len(image_paths) != len(document["inputs"]):
        raise ValueError("explicit image count differs from frozen inventory")
    sidecar = freeze_path.with_name(document["mask_sidecar"]["file"]).read_bytes()
    frames = []
    for image_path, frame, input_record in zip(image_paths, document["frames"], document["inputs"]):
        image = _load_pixels(image_path, input_record)
        height, width = image.shape[:2]
        usable = ~create_gui_mask(height, width)
        calibration = frame["calibration"]
        if calibration is None:
            frames.append({"frame_id": frame["frame_id"], "calibration": "abstained",
                           "candidate_count": 0, "reading_count": 0})
            continue
        axes = np.asarray(axis_directions(
            float(calibration["yaw_degrees"]), float(calibration["pitch_degrees"]),
            float(calibration["roll_degrees"])), dtype=float)
        principal = calibration["principal_point"]
        if principal is None:
            principal = (width / 2.0, height / 2.0)
        else:
            principal = tuple(float(value) for value in principal)
        candidates = []
        masks_by_identity = {}
        for index, proposal in enumerate(frame["proposal"]["proposals"]):
            mask = _mask_from_ref(sidecar, proposal["mask"]) & usable
            masks_by_identity[(proposal["axis"], json.dumps(proposal["polygon"], separators=(",", ":")))] = mask
            axis = int(proposal["axis"])
            window = _window_stats(mask, axis, axes, float(calibration["focal_pixels"]), principal)
            extent = window["stable_extent"]
            requested = (None if extent is None else
                         int(_legacy_sampling_for(extent, float(calibration["focal_pixels"]))))
            actual_capped = (None if requested is None else min(max(requested, 64), MAX_RECTIFIED_SIDE))
            candidates.append({
                "index": index, "axis": axis, "polygon": proposal["polygon"],
                **window,
                "initial_sampling_request": requested,
                "initial_sampling_after_cap": actual_capped,
                "would_hit_2048_initial_cap": bool(requested is not None and requested > MAX_RECTIFIED_SIDE),
            })
        readings = []
        for reading in frame["readings"]:
            identity = (reading["axis"], json.dumps(reading["polygon"], separators=(",", ":")))
            mask = masks_by_identity.get(identity)
            if mask is None:
                reads_mask = None
                reading_stats = {"period_recomputed": False, "reason": "proposal_mask_not_found"}
            else:
                reads_mask = mask
                reading_stats = _reading_diagnostics(image, usable, reading, mask)
            distances = [float(item["distance_blocks"]) for item in reading["alternatives"]]
            readings.append({
                "axis": int(reading["axis"]), "side": float(reading["side"]),
                "extent": float(reading["extent"]), "samples": int(reading["samples"]),
                "alternative_distances_blocks": distances,
                "alternative_supports": [float(item["support"]) for item in reading["alternatives"]],
                "all_alternatives_below_3_blocks": bool(distances and max(distances) < 3.0),
                "extent_over_1000": bool(reading["extent"] > 1000.0),
                "samples_at_2048_cap": bool(reading["samples"] == MAX_RECTIFIED_SIDE),
                **reading_stats,
            })
        frames.append({
            "frame_id": frame["frame_id"],
            "candidate_count": len(candidates),
            "reading_count": len(readings),
            "candidate_incidence": candidates,
            "readings": readings,
        })
    all_candidates = [item for frame in frames for item in frame.get("candidate_incidence", [])]
    all_readings = [item for frame in frames for item in frame.get("readings", [])]
    replay_errors = [float(item["period_relative_error"]) for item in all_readings
                     if item.get("period_recomputed") and
                     np.isfinite(item.get("period_relative_error", float("nan")))]
    return {
        "schema_version": 1,
        "diagnostic_version": 3,
        "labels_present": False,
        "diagnostic_source_sha256": _sha256(Path(__file__).read_bytes()),
        "freeze_sha256": _sha256(freeze_path.read_bytes()),
        "artifact_id": document["artifact_id"],
        "inputs": [{"basename": item["basename"], "width": item["width"],
                    "height": item["height"], "sha256": item["sha256"]}
                   for item in document["inputs"]],
        "config": {"grazing_incidence": GRAZING_INCIDENCE,
                   "legacy_plane_window_cut": 1e-6,
                   "excluded_pixel_fill": 0,
                   "max_rectified_side": MAX_RECTIFIED_SIDE,
                   "support_lag_multiple": 2,
                   "adaptive_rounds": 3,
                   "period_replay": "legacy adaptive loop with executed-sample trace"},
        "summary": {
            "frames": len(frames), "candidates": len(all_candidates),
            "readings": len(all_readings),
            "candidates_with_grazing_pixels": sum(item["grazing_pixels"] > 0 for item in all_candidates),
            "candidates_with_no_stable_incidence": sum(item["stable_pixels"] == 0 for item in all_candidates),
            "candidates_initial_sampling_over_2048": sum(item["would_hit_2048_initial_cap"] for item in all_candidates),
            "readings_all_distance_alternatives_below_3": sum(item["all_alternatives_below_3_blocks"] for item in all_readings),
            "readings_extent_over_1000": sum(item["extent_over_1000"] for item in all_readings),
            "readings_samples_at_2048": sum(item["samples_at_2048_cap"] for item in all_readings),
            "readings_with_period_recomputed": sum(item.get("period_recomputed", False) for item in all_readings),
            "readings_period_replay_over_5_percent": sum(error > 0.05 for error in replay_errors),
            "max_period_replay_relative_error": max(replay_errors, default=0.0),
            "readings_winning_vertical_profile": sum(item.get("winning_profile_direction") == 1 for item in all_readings),
        },
        "frames": frames,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--images", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("diagnostic output already exists; choose a fresh path")
    result = diagnose(args.freeze, args.images)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
