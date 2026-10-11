"""Independent binary64 appearance diagnostics, without state selection.

Feature and cost reductions are explicit left folds in C pixel/component order.
These synthetic diagnostics carry no calibration or physical-truth claim.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

from mc_photogrammetry.blocks.appearance import PaletteColor, palette_face_colors
from mc_photogrammetry.blocks.candidate_universe import record_sha256
from mc_photogrammetry.pipeline.image_derived_output_policy_v2 import ScorerIdentity

from .face_observation import FaceObservation


def scorer_definition() -> dict:
    """Fresh canonical built-in specification; callers cannot mutate the model."""
    return {
        "id": "exhaustive-appearance-diagnostics-v2", "version": "1",
        "domain": "producer-admitted immutable FaceObservation and complete PaletteColor tuple",
        "expected_rgb": (
            'palette_face_colors: float32 mapping, first mapped fallback, '
            'directional shade'
        ),
        "shade": {"down": 0.5, "up": 1.0, "north": 0.8, "south": 0.8,
                  "west": 0.6, "east": 0.6},
        "absolute": "(0.25*L2(rgb_o-rgb_c)+0.75*441.67*L2(chroma_o-chroma_c))+80*min_descriptor_L2",
        "chroma": "rgb/max(left_fold_sum(rgb),1e-6)",
        "descriptor_fallback": "empty candidate descriptor tuple adds exactly zero",
        "features": {
            "rgb": "binary64 C-order coverage*valid weighted mean; no float32 rounding",
            "descriptor": (
                '7 binary64 components: 4 red-chroma density bins, mean/255, '
                'std/255, vertical_gradient/255'
            ),
            "histogram": "red=R/max(R+G+B,1); bins=min(3,floor(4*red)); density=4*bin_weight/mass",
            "variance": "coverage*valid population variance of (R+G+B)/3",
            "gradient": (
                'unweighted vertical absolute luminance differences; excluded pixels '
                'replaced by weighted RGB mean'
            ),
        },
        "reduction": (
            'binary64 Python left fold; C-order pixels, RGB/components '
            'ascending, descriptors tuple order'
        ),
        "norm": "sqrt(left_fold_sum(component_difference_squared))",
        "no_evidence": "mass<=1e-6 or usable_coverage==0 or weight*usable_coverage==0",
        "relative": "absolute minus complete-universe minimum, independently per observation",
        "aggregation": (
            'canonical observation order left-fold '
            'sum(weight*usable_coverage*absolute)/sum(weight*usable_coverage); '
            'then subtract cell minimum'
        ),
        "zero": "all numerical signed zeros normalized positive",
        "ties": "equality of finite computed absolute binary64 values; partition all indices",
        "absent": ["clipping", "calibration", "light_fit", "prior", "shape", "simplicity",
                   "thumbnail_decode", "texture_correlation", "retrieval", "threshold",
                   "selection"],
        "failure": "any nonfinite feature or cost refuses complete producer request",
    }


def scorer_identity() -> ScorerIdentity:
    return ScorerIdentity(
        "exhaustive-appearance-diagnostics-v2", "1", "raw-rgb-chroma-descriptor-binary64-1",
        record_sha256(scorer_definition()),
        "absolute population invariant; relative complete minimum",
        "binary64 deterministic left folds; positive zero", "none",
    )


def finite(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("nonfinite feature or cost")
    return 0.0 if value == 0.0 else value


def cost_bits(value: float) -> str:
    """Canonical big-endian IEEE-754 binary64, exactly sixteen lowercase hex digits."""
    return struct.pack(">d", finite(value)).hex()


def cost_from_bits(value: str) -> float:
    if (type(value) is not str or len(value) != 16
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError("cost encoding must be sixteen lowercase hex digits")
    result = finite(struct.unpack(">d", bytes.fromhex(value))[0])
    if cost_bits(result) != value:
        raise ValueError("noncanonical signed zero")
    return result


def _sum(values) -> float:
    total = 0.0
    for value in values:
        total += float(value)
    return finite(total)


def _norm(values) -> float:
    return finite(math.sqrt(_sum(value * value for value in values)))


def tie_groups(absolute: tuple[float, ...]) -> tuple[tuple[int, ...], ...]:
    """Full exact equality partition, including singletons, in increasing cost order."""
    groups: dict[float, list[int]] = {}
    for index, value in enumerate(absolute):
        groups.setdefault(finite(value), []).append(index)
    return tuple(tuple(groups[value]) for value in sorted(groups))


@dataclass(frozen=True, slots=True)
class DiagnosticScores:
    absolute: tuple[float, ...]
    relative: tuple[float, ...]
    ties: tuple[tuple[int, ...], ...]


def diagnostic_scores(absolute: tuple[float, ...]) -> DiagnosticScores:
    if not absolute:
        raise ValueError("complete candidate vector cannot be empty")
    absolute = tuple(finite(value) for value in absolute)
    if any(value < 0 for value in absolute):
        raise ValueError("negative absolute cost")
    minimum = min(absolute)
    relative = tuple(finite(value - minimum) for value in absolute)
    return DiagnosticScores(absolute, relative, tie_groups(absolute))


@dataclass(frozen=True, slots=True)
class ObservationFeatures:
    mass: float
    fusion_weight: float
    rgb: tuple[float, ...] | None
    descriptor: tuple[float, ...] | None
    reason: str | None


def observation_features(observation: FaceObservation) -> ObservationFeatures:
    """Extract admitted features without population dependence or implicit reductions."""
    pixels = observation.patch.reshape(-1, 3)
    valid = None if observation.valid is None else observation.valid.reshape(-1)
    coverage = None if observation.coverage is None else observation.coverage.reshape(-1)
    weights = tuple((1.0 if coverage is None else float(coverage[index]))
                    if valid is None or bool(valid[index]) else 0.0
                    for index in range(len(pixels)))
    mass = _sum(weights)
    fusion_weight = finite(observation.weight * observation.usable_coverage)
    reason = ("zero_pixel_mass" if mass <= 1e-6 else
              "zero_usable_coverage" if observation.usable_coverage == 0 else
              "zero_fusion_weight" if fusion_weight == 0 else None)
    if reason is not None:
        return ObservationFeatures(mass, fusion_weight, None, None, reason)
    rgb = tuple(finite(_sum(float(pixel[channel]) * weight
                            for pixel, weight in zip(pixels, weights)) / mass)
                for channel in range(3))
    bins = [0.0] * 4
    luminances = []
    for pixel, weight in zip(pixels, weights):
        color = tuple(float(value) for value in pixel) if weight > 0 else rgb
        total = _sum(color)
        red = color[0] / max(total, 1.0)
        bins[min(3, int(math.floor(red * 4)))] += weight
        luminances.append(finite(total / 3.0))
    average = _sum(value * weight for value, weight in zip(luminances, weights)) / mass
    variance = _sum((value - average) ** 2 * weight
                    for value, weight in zip(luminances, weights)) / mass
    width = observation.patch.shape[1]
    gradient = _sum(abs(luminances[index] - luminances[index - width])
                    for index in range(width, len(luminances))) / (len(luminances) - width)
    descriptor = tuple(finite(value) for value in (
        *(4.0 * value / mass for value in bins), average / 255.0,
        math.sqrt(variance) / 255.0, gradient / 255.0,
    ))
    return ObservationFeatures(mass, fusion_weight, rgb, descriptor, None)


class AppearanceScorerV2:
    """Fixed scorer; production entry is the bounded diagnostic producer only."""

    def __init__(self, palette: tuple[PaletteColor, ...]) -> None:
        if (type(palette) is not tuple or not palette
                or any(type(item) is not PaletteColor for item in palette)):
            raise ValueError("scorer requires an admitted nonempty palette tuple")
        self._palette = palette
        self._colors = palette_face_colors(palette)

    def score(self, features: ObservationFeatures, slot: str) -> DiagnosticScores:
        if features.reason is not None or features.rgb is None or features.descriptor is None:
            raise ValueError("no evidence must not be scored")
        rgb, descriptor = features.rgb, features.descriptor
        observed_total = max(_sum(rgb), 1e-6)
        absolute = []
        for item in self._palette:
            expected = tuple(float(value) for value in self._colors[item.state][slot])
            expected_total = max(_sum(expected), 1e-6)
            straight = _norm(a - b for a, b in zip(rgb, expected))
            chroma = _norm(a / observed_total - b / expected_total
                           for a, b in zip(rgb, expected))
            descriptor_cost = (min(_norm(a - float(b) for a, b in zip(descriptor, candidate))
                                   for candidate in item.descriptors)
                               if item.descriptors else 0.0)
            absolute.append(finite((0.25 * straight + 0.75 * 441.67 * chroma)
                                   + 80.0 * descriptor_cost))
        return diagnostic_scores(tuple(absolute))


def aggregate_scores(rows: tuple[tuple[float, DiagnosticScores], ...]) -> DiagnosticScores:
    """Aggregate absolute vectors; relative observation vectors never enter fusion."""
    if not rows or any(weight <= 0 for weight, _ in rows):
        raise ValueError("aggregation requires usable positive-weight evidence")
    count = len(rows[0][1].absolute)
    if any(len(scores.absolute) != count for _, scores in rows):
        raise ValueError("incomplete aggregate vector")
    mass = _sum(weight for weight, _ in rows)
    return diagnostic_scores(tuple(
        finite(_sum(weight * scores.absolute[index] for weight, scores in rows) / mass)
        for index in range(count)
    ))
