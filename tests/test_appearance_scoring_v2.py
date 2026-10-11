"""Synthetic scalar references for the independent diagnostic model."""

from __future__ import annotations

import math
import struct
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from mc_photogrammetry.blocks.appearance import PaletteColor, palette_face_colors
from mc_photogrammetry.domain.models import BlockState
from mc_photogrammetry.reconstruction import appearance_scoring_v2 as subject
from mc_photogrammetry.reconstruction.face_observation import NORMAL_OF_SLOT, FaceObservation


def observation(patch, *, slot="up", valid=None, coverage=None, weight=1.0, usable=1.0):
    return FaceObservation((0, 0, 0), slot, "frame", NORMAL_OF_SLOT[slot],
                           np.asarray(patch, dtype=np.float32), usable, ("frame",),
                           valid, coverage, ("synthetic",), weight)


def candidate(name, rgb, descriptors=(), face_map=(), face_rgbs=()):
    return PaletteColor(BlockState("minecraft:" + name), rgb, face_rgbs=face_rgbs,
                        descriptors=descriptors, face_map=face_map)


def scalar_norm(values):
    total = 0.0
    for value in values:
        total += value * value
    return math.sqrt(total)


def scalar_reference(rgb, descriptor, expected, candidate_descriptors):
    observed_sum = 0.0
    expected_sum = 0.0
    for a, b in zip(rgb, expected):
        observed_sum += a
        expected_sum += float(b)
    rgb_gap = scalar_norm(a - float(b) for a, b in zip(rgb, expected))
    chroma_gap = scalar_norm(a / max(observed_sum, 1e-6) - float(b) / max(expected_sum, 1e-6)
                             for a, b in zip(rgb, expected))
    desc_gap = (min(scalar_norm(a - b for a, b in zip(descriptor, row))
                    for row in candidate_descriptors) if candidate_descriptors else 0.0)
    return (0.25 * rgb_gap + 0.75 * 441.67 * chroma_gap) + 80.0 * desc_gap


@pytest.mark.parametrize("slot", ("down", "up", "north", "south", "east", "west"))
def test_every_candidate_cost_equals_scalar_binary64_reference(slot):
    palette = (
        candidate("a", (61, 72, 83)),
        candidate("b", (200, 10, 30), descriptors=((1., 2., 3., 4., .5, .7, .1),)),
        candidate("c", (2, 2, 2), descriptors=((0.,) * 7, (2.,) * 7),
                  face_map=(("north", (90, 81, 12)), ("up", (10, 100, 40)))),
    )
    patch = np.arange(18, dtype=np.float32).reshape(2, 3, 3) * 10
    features = subject.observation_features(observation(patch, slot=slot))
    scored = subject.AppearanceScorerV2(palette).score(features, slot)
    colors = palette_face_colors(palette)
    for index, item in enumerate(palette):
        reference = scalar_reference(features.rgb, features.descriptor,
                                     colors[item.state][slot], item.descriptors)
        assert subject.cost_bits(scored.absolute[index]) == struct.pack(">d", reference).hex()
    assert scored.relative == tuple(value - min(scored.absolute) for value in scored.absolute)


def test_absolute_population_invariance_and_unclipped_relative_costs():
    a = candidate("a", (100, 100, 100))
    b = candidate("b", (255, 0, 0))
    c = candidate("c", (0, 255, 0))
    features = subject.observation_features(observation(np.full((2, 2, 3), 100)))
    first = subject.AppearanceScorerV2((a, b)).score(features, "up")
    extended = subject.AppearanceScorerV2((c, b, a)).score(features, "up")
    assert first.absolute == (extended.absolute[2], extended.absolute[1])
    assert first.relative[1] > 24.0


def test_weighted_absolute_aggregation_scalar_reference_and_relative_derivation():
    rows = ((.25, subject.diagnostic_scores((1.0, 8.0, 8.0))),
            (3.5, subject.diagnostic_scores((9.0, 2.0, 2.0))))
    scores = subject.aggregate_scores(rows)
    reference = tuple((.25 * rows[0][1].absolute[i] + 3.5 * rows[1][1].absolute[i]) / 3.75
                      for i in range(3))
    assert scores.absolute == reference
    assert scores.relative == tuple(value - min(reference) for value in reference)
    averaged_relatives = tuple((.25 * rows[0][1].relative[i] + 3.5 * rows[1][1].relative[i]) / 3.75
                              for i in range(3))
    assert scores.relative != averaged_relatives
    assert scores.ties == ((1, 2), (0,))


def test_masked_features_shading_and_empty_descriptor_fallback():
    patch = np.array([[[100, 60, 20], [255, 255, 255]],
                      [[100, 60, 20], [0, 0, 0]]], dtype=np.float32)
    valid = np.array([[True, False], [True, False]])
    coverage = np.array([[.25, 1], [.75, 1]], dtype=np.float64)
    features = subject.observation_features(observation(patch, valid=valid, coverage=coverage))
    assert features.rgb == (100., 60., 20.)
    assert features.mass == 1.0
    assert features.descriptor == (0., 0., 4., 0., 60. / 255., 0., 0.)
    item = candidate("a", (200, 120, 40))
    scored = subject.AppearanceScorerV2((item,)).score(features, "down")
    assert scored.absolute == (0.0,)
    assert scored.ties == ((0,),)


def test_mapping_and_fallback_follow_palette_face_colors_exactly():
    palette = (candidate("a", (255, 0, 0), face_rgbs=((0, 200, 0),)),
               candidate("b", (0, 0, 0), face_map=(("north", (100, 100, 100)),)))
    features = subject.observation_features(observation(np.full((2, 1, 3), 60)))
    scores = subject.AppearanceScorerV2(palette).score(features, "east")
    assert scores.absolute[1] == pytest.approx(0.0, abs=1e-5)
    assert scores.absolute[0] > 0


@pytest.mark.parametrize("mode", ("zero_mass", "epsilon_mass", "zero_usable", "zero_weight"))
def test_no_evidence_is_explicit_without_scores(mode):
    coverage = np.full((2, 1), 0 if mode == "zero_mass" else 5e-7, dtype=np.float64)
    obs = observation(np.full((2, 1, 3), 70),
                      coverage=coverage if "mass" in mode else None,
                      weight=0 if mode == "zero_weight" else 1,
                      usable=0 if mode == "zero_usable" else 1)
    features = subject.observation_features(obs)
    assert features.reason is not None
    assert features.rgb is None and features.descriptor is None
    with pytest.raises(ValueError, match="no evidence"):
        subject.AppearanceScorerV2((candidate("a", (70, 70, 70)),)).score(features, "up")


def test_distinct_identical_appearances_preserve_full_exact_ties_and_immutability():
    palette = tuple(candidate(name, (40, 40, 40)) for name in ("a", "b", "c"))
    features = subject.observation_features(observation(np.full((2, 1, 3), 40)))
    scores = subject.AppearanceScorerV2(palette).score(features, "up")
    assert scores.absolute == (0.0, 0.0, 0.0)
    assert scores.ties == ((0, 1, 2),)
    with pytest.raises(FrozenInstanceError):
        scores.absolute = ()


@pytest.mark.parametrize("value", (0.0, -0.0, 1.0, 1e-100, 1e100, math.nextafter(1., 2.)))
def test_exact_cost_binary64_replay_and_zero_normalization(value):
    encoded = subject.cost_bits(value)
    assert len(encoded) == 16
    assert subject.cost_from_bits(encoded) == value
    assert encoded == struct.pack(">d", 0. if value == 0 else value).hex()


@pytest.mark.parametrize("encoded", ("8000000000000000", "7ff0000000000000",
                                    "7ff8000000000000", "3FF0000000000000", "0"))
def test_invalid_or_noncanonical_cost_bits_refuse(encoded):
    with pytest.raises(ValueError):
        subject.cost_from_bits(encoded)


def test_nonfinite_feature_or_cost_refuses_and_definition_is_not_mutable():
    with pytest.raises(ValueError):
        subject.diagnostic_scores((math.inf,))
    features = subject.ObservationFeatures(2., 1., (math.nan, 0., 0.), (0.,) * 7, None)
    with pytest.raises(ValueError):
        subject.AppearanceScorerV2((candidate("a", (0, 0, 0)),)).score(features, "up")
    identity = subject.scorer_identity()
    definition = subject.scorer_definition()
    definition["features"]["rgb"] = "forged"
    assert subject.scorer_identity() == identity
