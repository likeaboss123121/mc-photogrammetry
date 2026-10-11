"""Reusable exact-state appearance scorer over typed face observations.

The scorer turns a palette of :class:`~mc_photogrammetry.blocks.appearance.PaletteColor`
records into a vectorised cost for every candidate state, then scores a
:class:`~mc_photogrammetry.reconstruction.face_observation.FaceObservation`
against it. It keeps shape and material verdicts separate: the optional shape
term only expresses how much of the observed face a candidate's model fills, and
it never decides occupancy.

This module intentionally depends on the neutral appearance records in
``mc_photogrammetry.blocks.appearance`` rather than on the legacy
``reconstruction.facade`` records or the ``reconstruction.raycast``
implementation, so the later M4/M5 adapter can consume the scorer without the
legacy facade.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mc_photogrammetry.blocks.appearance import (
    CHROMA_WEIGHT,
    AppearanceCalibrationLike,
    PaletteColor,
    appearance_distance,
    normalised_pattern,
    palette_face_colors,
    slot_coverage,
)
from mc_photogrammetry.blocks.model_atlas import FACE_SLOTS
from mc_photogrammetry.domain.models import BlockState
from .face_observation import FaceObservation
from .face_sampling import PATCH_SIZE

MINIMUM_LIGHT_FACTOR = 0.15
"""Darkest a face may be assumed lit when undoing lighting.

Dividing by anything smaller turns sensor noise in a nearly black patch
into a confident colour.
"""

OUTLIER_CEILING = 24.0
"""Cap on how much worse than its own best match a candidate may score in one view."""


def palette_face_textures(
    palette: tuple[PaletteColor, ...],
    *,
    patch_size: int = PATCH_SIZE,
) -> dict[str, tuple[list[BlockState], np.ndarray]]:
    """Per face slot, the candidate states and their normalised texture patterns.

    Each texture is reduced to zero-mean, unit-norm luminance, so a dot product
    against a patch prepared the same way is the correlation between them. That
    is invariant to how brightly the face happened to be lit, which is the one
    thing block lighting changes most and material identity changes least.
    """
    import base64
    from io import BytesIO

    per_slot: dict[str, tuple[list[BlockState], list[np.ndarray]]] = {}
    for item in palette:
        for slot, encoded in item.face_thumbnails:
            try:
                from PIL import Image

                pixels = np.asarray(
                    Image.open(BytesIO(base64.b64decode(encoded))), dtype=np.float32
                )
            except (ValueError, OSError, TypeError, ImportError):
                continue
            if pixels.ndim != 3 or pixels.shape[:2] != (patch_size, patch_size):
                continue
            vector = normalised_pattern(pixels)
            if vector is None:
                continue
            states, vectors = per_slot.setdefault(slot, ([], []))
            states.append(item.state)
            vectors.append(vector)
    return {
        slot: (states, np.stack(vectors)) for slot, (states, vectors) in per_slot.items() if vectors
    }


class PaletteIndex:
    """Palette laid out as arrays so one observation is scored in a few array ops.

    Scoring walked every candidate in Python, which at several thousand faces and
    several hundred candidates was tens of millions of three-element numpy calls
    and most of a reconstruction's runtime. Everything a candidate contributes -
    its colour per face slot, its texture descriptors, its family prior, its
    texture pattern - is stacked here once, so a face costs a handful of whole
    array operations instead.
    """

    def __init__(
        self,
        palette: tuple[PaletteColor, ...],
        face_colors: dict[BlockState, dict[str, np.ndarray]],
        family_priors: dict[str, float] | None,
        texture_weight: float = 0.0,
    ) -> None:
        self.states: list[BlockState] = [item.state for item in palette]
        self.count = len(self.states)
        self.colours = {
            slot: np.stack([face_colors[item.state][slot] for item in palette]).astype(np.float64)
            for slot in FACE_SLOTS
        }
        rows: list[tuple[float, ...]] = []
        owners: list[int] = []
        for index, item in enumerate(palette):
            for descriptor in item.descriptors:
                rows.append(descriptor)
                owners.append(index)
        self.descriptors = np.asarray(rows, dtype=np.float64) if rows else None
        self.descriptor_owner = np.asarray(owners, dtype=np.intp) if rows else None
        self.priors = (
            np.asarray([max(0.1, family_priors.get(item.family, 1.0)) for item in palette])
            if family_priors
            else None
        )
        # Among candidates that explain a face equally well, prefer the one whose
        # appearance is simplest: the number of distinct faces it wears. Plain
        # planks show one texture, a log two, a workstation three or more. A wall
        # seen only side-on cannot tell dark oak planks from a cartography table,
        # which carries the same wood on its sides and a map on top, and the
        # planks are the simpler explanation of what was actually seen.
        # Plain building material wears one face (planks, stone) or two (a log's
        # bark and end, a blackstone's top and side). Three or more distinct
        # faces means a fixture - a workstation, a furnace, an observer - whose
        # sides borrow an ordinary material. A wall seen only side-on cannot tell
        # dark oak planks from a cartography table wearing the same wood, and the
        # planks are the simpler explanation of what was actually seen. Counting
        # only faces beyond the second leaves the log-versus-planks call, which
        # real texture evidence settles, untouched.
        distinct = np.asarray(
            [len({tuple(colour) for _slot, colour in item.face_map}) or 1 for item in palette],
            dtype=np.float64,
        )
        self.simplicity = np.maximum(distinct - 2.0, 0.0)
        # How much of each cell face every candidate's shape actually fills, so a
        # partial block is not asked to explain a face it only partly occupies.
        self.coverage = {
            slot: np.array([slot_coverage(item, slot) for item in palette], dtype=np.float64)
            for slot in FACE_SLOTS
        }
        self.prior: np.ndarray | None = None
        self.textures: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if texture_weight > 0.0:
            position = {state: index for index, state in enumerate(self.states)}
            for slot, (states, matrix) in palette_face_textures(palette).items():
                mask = np.zeros(self.count, dtype=bool)
                aligned = np.zeros((self.count, matrix.shape[1]), dtype=np.float32)
                for row, state in enumerate(states):
                    index = position.get(state)
                    if index is not None:
                        aligned[index] = matrix[row]
                        mask[index] = True
                self.textures[slot] = (aligned, mask)

    def with_prior(self, prior: np.ndarray) -> "PaletteIndex":
        """A view of this index carrying an additive per-candidate cost."""
        clone = object.__new__(PaletteIndex)
        clone.__dict__.update(getattr(self, "__dict__", {}))
        for name in ("states", "count", "colours", "descriptors", "descriptor_owner",
                     "priors", "textures", "simplicity", "prior"):
            setattr(clone, name, getattr(self, name, None))
        clone.prior = np.asarray(prior, dtype=np.float64)
        return clone

    def restricted(self, keep: np.ndarray) -> "PaletteIndex":
        """A view of this index holding only the given candidate indices."""
        clone = object.__new__(PaletteIndex)
        clone.states = [self.states[int(i)] for i in keep]
        clone.count = len(clone.states)
        clone.colours = {slot: value[keep] for slot, value in self.colours.items()}
        if self.descriptors is not None:
            position = {int(old): new for new, old in enumerate(keep)}
            rows, owners = [], []
            for row, owner in enumerate(self.descriptor_owner):
                moved = position.get(int(owner))
                if moved is not None:
                    rows.append(self.descriptors[row])
                    owners.append(moved)
            clone.descriptors = np.asarray(rows) if rows else None
            clone.descriptor_owner = np.asarray(owners, dtype=np.intp) if rows else None
        else:
            clone.descriptors = None
            clone.descriptor_owner = None
        clone.priors = None if self.priors is None else self.priors[keep]
        clone.simplicity = self.simplicity[keep]
        clone.coverage = {slot: values[keep] for slot, values in self.coverage.items()}
        clone.textures = {
            slot: (aligned[keep], mask[keep]) for slot, (aligned, mask) in self.textures.items()
        }
        return clone

    def costs(
        self,
        rgb: np.ndarray,
        descriptor: tuple[float, ...],
        slot: str,
        *,
        calibration: AppearanceCalibrationLike | None = None,
        light: float = 1.0,
        pattern: np.ndarray | None = None,
        texture_weight: float = 0.0,
        outlier_ceiling: float = OUTLIER_CEILING,
        simplicity_weight: float = 0.0,
        shape_weight: float = 0.0,
    ) -> tuple[np.ndarray, float]:
        """Relative, capped cost of every candidate for one observed face."""
        # Undo the lighting rather than dim every candidate to match it. Dividing
        # the observation back up to how the block would look at full brightness
        # keeps every face on the same footing as the jar's own textures: a face
        # in shadow then has the same room to be wrong as a lit one, where
        # dimming the expectation squeezes the candidates together and makes a
        # dark face weak evidence for anything.
        if light < 1.0:
            rgb = rgb / max(light, MINIMUM_LIGHT_FACTOR)
        rgb = np.asarray(rgb, dtype=np.float64)
        expected = self.colours[slot]
        if calibration is not None:
            expected = expected * np.asarray(calibration.scale) + np.asarray(calibration.offset)
        difference = rgb - expected
        straight = np.sqrt(np.einsum("ij,ij->i", difference, difference))
        observed_total = max(float(rgb.sum()), 1e-6)
        expected_total = np.maximum(expected.sum(axis=1, keepdims=True), 1e-6)
        chroma = rgb / observed_total - expected / expected_total
        chroma = np.sqrt(np.einsum("ij,ij->i", chroma, chroma)) * 441.67
        distance = (1.0 - CHROMA_WEIGHT) * straight + CHROMA_WEIGHT * chroma
        if self.descriptors is not None and descriptor:
            gap = self.descriptors - np.asarray(descriptor, dtype=np.float64)
            gap = np.sqrt(np.einsum("ij,ij->i", gap, gap))
            closest = np.full(self.count, np.inf)
            np.minimum.at(closest, self.descriptor_owner, gap)
            distance = distance + 80.0 * np.where(np.isfinite(closest), closest, 0.0)
        if self.priors is not None:
            distance = distance / self.priors
        if pattern is not None and texture_weight > 0.0:
            entry = self.textures.get(slot)
            if entry is not None:
                aligned, mask = entry
                correlation = aligned[mask] @ pattern
                if correlation.size:
                    # Score each candidate against how well a typical candidate
                    # explains this patch, not against a perfect match. Real
                    # screenshots are far blurrier than the 16-pixel source
                    # textures, so every correlation is small and an absolute
                    # penalty charges nearly the full weight to everyone -
                    # including, through the fixed fallback, the candidates with
                    # no pattern for this slot, which were meant to be neutral.
                    # Centring makes the term zero-sum: it only ever expresses
                    # which candidate matches better than the rest, and a patch
                    # too blurred to separate them contributes nothing.
                    neutral = float(correlation.mean())
                    penalty = np.zeros(self.count)
                    penalty[mask] = texture_weight * (neutral - correlation)
                    distance = distance + penalty
        if simplicity_weight > 0.0:
            distance = distance + simplicity_weight * self.simplicity
        if shape_weight > 0.0:
            covered = self.coverage.get(slot)
            if covered is not None:
                distance = distance + shape_weight * (1.0 - covered)
        prior = getattr(self, "prior", None)
        if prior is not None:
            distance = distance + prior
        best = float(distance.min())
        return np.minimum(distance - best, outlier_ceiling), best


def observation_costs(
    rgb: np.ndarray,
    descriptor: tuple[float, ...],
    slot: str,
    palette: tuple[PaletteColor, ...],
    face_colors: dict[BlockState, dict[str, np.ndarray]],
    calibration: AppearanceCalibrationLike | None,
    family_priors: dict[str, float] | None,
    slot_matching: bool,
    outlier_ceiling: float = OUTLIER_CEILING,
    light: float = 1.0,
    pattern: np.ndarray | None = None,
    face_textures: dict[str, tuple[list[BlockState], np.ndarray]] | None = None,
    texture_weight: float = 0.0,
) -> tuple[dict[BlockState, float], float]:
    """Score one observation against every palette entry.

    Costs are returned relative to the observation's own best match and capped.
    A sample that landed on the wrong block - because a camera pose is slightly
    off, or something outside the reconstruction occludes it - fits nothing well,
    so every candidate hits the ceiling and the observation abstains instead of
    voting confidently for the wrong answer. The absolute best cost comes back
    too, because it says how much this observation should be trusted at all.
    """
    # Undo the lighting rather than dim every candidate to match it. Dividing the
    # observation back up to how the block would look at full brightness keeps
    # every face on the same footing as the jar's own textures: a face in shadow
    # then has the same room to be wrong as a lit one, where dimming the
    # expectation instead squeezes all the candidates together and makes a dark
    # face weak evidence for anything.
    if light < 1.0:
        rgb = rgb / max(light, MINIMUM_LIGHT_FACTOR)
    texture_cost: dict[BlockState, float] = {}
    if pattern is not None and face_textures and texture_weight > 0.0:
        entry = face_textures.get(slot)
        if entry is not None:
            states, matrix = entry
            correlation = matrix @ pattern
            texture_cost = {
                state: texture_weight * (1.0 - float(value))
                for state, value in zip(states, correlation)
            }
    distances: dict[BlockState, float] = {}
    for item in palette:
        faces = face_colors[item.state]
        candidates = (faces[slot],) if slot_matching and slot in faces else tuple(faces.values())
        if not candidates:
            continue
        distance = min(
            appearance_distance(
                rgb,
                calibration.apply(expected) if calibration else expected,
            )
            for expected in candidates
        )
        if item.descriptors:
            distance += 80.0 * min(
                float(np.linalg.norm(np.asarray(descriptor) - np.asarray(candidate)))
                for candidate in item.descriptors
            )
        if family_priors:
            distance /= max(0.1, family_priors.get(item.family, 1.0))
        # A flat candidate has no pattern to match, so it neither gains nor loses.
        distance += texture_cost.get(item.state, texture_weight)
        distances[item.state] = distance
    if not distances:
        return {}, float("inf")
    best = min(distances.values())
    return {state: min(distance - best, outlier_ceiling) for state, distance in distances.items()}, best


def _candidate_sort_key(state: BlockState) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Canonical, palette-independent ordering key for a block state.

    ``BlockState._identity`` is the exact normalised value equality uses, so two
    spellings of the same state sort identically.
    """
    identity = state._identity
    if identity is None:  # pragma: no cover - always set by BlockState.__post_init__
        return (state.name, tuple(sorted(state.properties)))
    return identity


@dataclass(frozen=True, slots=True)
class FaceScore:
    """Relative state costs for one face observation, or a typed abstention."""

    costs: tuple[tuple[BlockState, float], ...]
    best_cost: float
    abstained: bool = False
    reason: str | None = None

    def cost_for(self, state: BlockState) -> float | None:
        for candidate, cost in self.costs:
            if candidate == state:
                return cost
        return None

    def ranked(self) -> tuple[tuple[BlockState, float], ...]:
        """Candidates from cheapest to dearest, ties in canonical state order.

        The tie-break is deliberately independent of palette order: two palettes
        that hold the same candidates in a different order must still nominate
        the same winner.
        """
        return tuple(
            sorted(
                self.costs,
                key=lambda item: (float(item[1]), _candidate_sort_key(item[0])),
            )
        )


def score_face_observation(
    observation: FaceObservation,
    index: PaletteIndex,
    *,
    calibration: AppearanceCalibrationLike | None = None,
    light: float = 1.0,
    texture_weight: float = 0.0,
    outlier_ceiling: float = OUTLIER_CEILING,
    simplicity_weight: float = 0.0,
    shape_weight: float = 0.0,
    use_pattern: bool = False,
) -> FaceScore:
    """Score a typed face observation against every palette candidate.

    A malformed or zero-coverage observation abstains with a typed reason rather
    than voting. Only pixels the observation marks valid and owned contribute:
    the patch's own excluded pixels never enter the colour, descriptor or
    texture pattern.
    """
    if index.count == 0:
        return FaceScore((), float("inf"), abstained=True, reason="empty_palette")
    if not observation.is_usable:
        return FaceScore((), float("inf"), abstained=True, reason=observation.abstention_reason)
    rgb = observation.rgb
    descriptor = observation.descriptor
    pattern = observation.pattern if use_pattern else None
    costs, best = index.costs(
        rgb,
        descriptor,
        observation.slot,
        calibration=calibration,
        light=light,
        pattern=pattern,
        texture_weight=texture_weight,
        outlier_ceiling=outlier_ceiling,
        simplicity_weight=simplicity_weight,
        shape_weight=shape_weight,
    )
    return FaceScore(
        tuple((state, float(cost)) for state, cost in zip(index.states, costs)),
        float(best),
    )


class AppearanceScorer:
    """A palette laid out once, ready to score typed face observations."""

    def __init__(
        self,
        palette: tuple[PaletteColor, ...],
        *,
        family_priors: dict[str, float] | None = None,
        texture_weight: float = 0.0,
        patch_size: int = PATCH_SIZE,
    ) -> None:
        if not palette:
            raise ValueError("palette cannot be empty")
        self.palette = palette
        self.texture_weight = texture_weight
        self.patch_size = patch_size
        self.index = PaletteIndex(
            palette, palette_face_colors(palette), family_priors, texture_weight
        )

    def score(self, observation: FaceObservation, **kwargs: object) -> FaceScore:
        return score_face_observation(
            observation, self.index, texture_weight=self.texture_weight, **kwargs  # type: ignore[arg-type]
        )

    def scores_for_slot(
        self,
        observations: tuple[FaceObservation, ...],
        *,
        slot: str | None = None,
    ) -> tuple[FaceScore, ...]:
        return tuple(
            self.score(observation)
            for observation in observations
            if slot is None or observation.slot == slot
        )


__all__ = [
    "AppearanceScorer",
    "FaceScore",
    "MINIMUM_LIGHT_FACTOR",
    "OUTLIER_CEILING",
    "PaletteIndex",
    "observation_costs",
    "palette_face_textures",
    "score_face_observation",
]
