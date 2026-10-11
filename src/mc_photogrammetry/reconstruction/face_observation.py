"""Immutable, validated world-face observations.

An :class:`FaceObservation` is one rectified look at one cell's world face from
one supporting frame. It is the typed seam between image-supported geometry
(M4/V2 placements) and material scoring (M5): geometry owns the cell, the world
face slot and the outward normal; the observation owns the retained pixels and
their provenance. It carries no occupancy verdict.

The contract is deliberately strict:

* the record is frozen and the retained arrays are copied and marked read-only,
  so no later mutation can change a score that was already computed from it;
* ``(cell, slot, frame_id)`` is the identity, and the outward ``normal`` must
  agree with the slot's canonical axis;
* every retained number is finite, coverage is bounded to ``[0, 1]``, and
  provenance references are non-empty unique strings;
* invalid or zero-coverage observations abstain instead of voting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mc_photogrammetry.blocks.appearance import normalised_pattern, patch_descriptor
from mc_photogrammetry.blocks.model_atlas import FACE_BASIS, FACE_SLOTS
from mc_photogrammetry.domain.models import BlockCoordinate

NORMAL_OF_SLOT: dict[str, tuple[float, float, float]] = {
    slot: tuple(float(value) for value in basis[0]) for slot, basis in FACE_BASIS.items()
}
"""Canonical outward world normal of each world-face slot."""

_REASONS = {0: "zero_coverage", 1: "no_valid_pixels"}


def _readonly(array: np.ndarray) -> np.ndarray:
    """A private, C-contiguous copy that no caller can make writeable again.

    An array that owns its data can always be reopened with
    ``setflags(write=True)``, so merely clearing the WRITEABLE flag is not
    enough. The copy is therefore backed by immutable ``bytes``: a view over a
    ``bytes`` buffer can never be made writeable, directly or through its base.
    """
    copy = np.array(array, copy=True, order="C")
    frozen = np.frombuffer(copy.tobytes(order="C"), dtype=copy.dtype).reshape(copy.shape)
    frozen.setflags(write=False)
    return frozen


@dataclass(frozen=True, eq=False, slots=True)
class FaceObservation:
    """One frame's evidence for one world face of one cell."""

    cell: BlockCoordinate
    slot: str
    frame_id: str
    normal: tuple[float, float, float]
    patch: np.ndarray
    """``(height, width, 3)`` rectified RGB, retained as a read-only copy."""
    usable_coverage: float
    """Share of the face that is usable evidence, in ``[0, 1]``."""
    support_frames: tuple[str, ...] = ()
    """Stable frame identifiers supporting this face; must include ``frame_id``."""
    valid: np.ndarray | None = None
    """Boolean ``(height, width)`` per-pixel ownership mask, copied read-only."""
    coverage: np.ndarray | None = None
    """Fractional ``(height, width)`` ownership per pixel in ``[0, 1]``, copied read-only."""
    provenance: tuple[str, ...] = ()
    """Stable references (camera solution, region evidence) for this observation."""
    weight: float = 1.0
    """Trust weight for fusion; a diagnostic, not a calibrated probability."""

    def __post_init__(self) -> None:
        cell = self.cell
        if (
            not isinstance(cell, tuple)
            or len(cell) != 3
            or any(isinstance(value, bool) or not isinstance(value, int) for value in cell)
        ):
            raise ValueError("cell must be a 3-tuple of integers")
        if self.slot not in FACE_SLOTS:
            raise ValueError(f"slot must be one of {FACE_SLOTS}")
        if not isinstance(self.frame_id, str) or not self.frame_id:
            raise ValueError("frame_id must be a non-empty string")

        normal = self.normal
        if not isinstance(normal, tuple) or len(normal) != 3:
            raise ValueError("normal must be a 3-tuple")
        normal_values = np.asarray(normal, dtype=np.float64)
        if not np.isfinite(normal_values).all():
            raise ValueError("normal must be finite")
        if abs(float(np.linalg.norm(normal_values)) - 1.0) > 1e-6:
            raise ValueError("normal must be a unit vector")
        canonical = np.asarray(NORMAL_OF_SLOT[self.slot], dtype=np.float64)
        if not np.allclose(normal_values, canonical, atol=1e-6):
            raise ValueError(f"normal is inconsistent with slot {self.slot!r}")

        patch = self.patch
        if (
            not isinstance(patch, np.ndarray)
            or patch.ndim != 3
            or patch.shape[2] != 3
            or patch.shape[0] < 1
            or patch.shape[1] < 1
        ):
            raise ValueError("patch must be a non-empty (height, width, 3) array")
        retained_patch = _readonly(np.asarray(patch, dtype=np.float32))
        if not np.isfinite(retained_patch).all():
            raise ValueError("patch must be finite")

        retained_valid: np.ndarray | None = None
        if self.valid is not None:
            valid = self.valid
            if not isinstance(valid, np.ndarray) or valid.dtype != np.bool_:
                raise ValueError("valid must be a boolean array")
            if valid.shape != retained_patch.shape[:2]:
                raise ValueError("valid must match the patch height and width")
            retained_valid = _readonly(valid)

        retained_coverage: np.ndarray | None = None
        if self.coverage is not None:
            coverage = self.coverage
            if not isinstance(coverage, np.ndarray) or coverage.shape != retained_patch.shape[:2]:
                raise ValueError("coverage must match the patch height and width")
            retained_coverage = _readonly(np.asarray(coverage, dtype=np.float64))
            if not np.isfinite(retained_coverage).all():
                raise ValueError("coverage must be finite")
            if float(retained_coverage.min()) < -1e-9 or float(retained_coverage.max()) > 1.0 + 1e-9:
                raise ValueError("coverage must lie in [0, 1]")

        usable = self.usable_coverage
        if isinstance(usable, bool) or not isinstance(usable, (int, float, np.floating)):
            raise ValueError("usable_coverage must be a number")
        usable = float(usable)
        if not np.isfinite(usable) or usable < -1e-9 or usable > 1.0 + 1e-9:
            raise ValueError("usable_coverage must lie in [0, 1]")

        support = self.support_frames
        if not isinstance(support, tuple) or not support or len(set(support)) != len(support):
            raise ValueError("support_frames must be a non-empty tuple of unique frame ids")
        if any(not isinstance(name, str) or not name for name in support):
            raise ValueError("support_frames entries must be non-empty strings")
        if self.frame_id not in support:
            raise ValueError("support_frames must include frame_id")

        provenance = self.provenance
        if not isinstance(provenance, tuple) or len(set(provenance)) != len(provenance):
            raise ValueError("provenance must be a tuple of unique references")
        if any(not isinstance(name, str) or not name for name in provenance):
            raise ValueError("provenance entries must be non-empty strings")

        weight = self.weight
        if isinstance(weight, bool) or not isinstance(weight, (int, float, np.floating)):
            raise ValueError("weight must be a number")
        weight = float(weight)
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError("weight must be finite and non-negative")

        object.__setattr__(self, "cell", (int(cell[0]), int(cell[1]), int(cell[2])))
        object.__setattr__(self, "normal", (float(normal[0]), float(normal[1]), float(normal[2])))
        object.__setattr__(self, "patch", retained_patch)
        object.__setattr__(self, "valid", retained_valid)
        object.__setattr__(self, "coverage", retained_coverage)
        object.__setattr__(self, "usable_coverage", min(1.0, max(0.0, usable)))
        object.__setattr__(self, "weight", weight)

    @property
    def identity(self) -> tuple[BlockCoordinate, str, str]:
        """The per-cell/per-slot/per-frame identity of this observation."""
        return (self.cell, self.slot, self.frame_id)

    @property
    def has_valid_pixels(self) -> bool:
        return self.valid is None or bool(self.valid.any())

    @property
    def has_effective_coverage(self) -> bool:
        """Whether any retained pixel carries positive effective weight.

        ``usable_coverage`` is a summary the sampler reports, not a substitute
        for the per-pixel masks: a face whose valid/coverage masks exclude every
        pixel is not observed no matter how positive that summary is.
        """
        effective = self.effective_coverage
        return effective is None or bool(np.any(effective > 0.0))

    @property
    def abstention_reason(self) -> str | None:
        """Why this observation cannot vote, or ``None`` when it is usable."""
        if self.usable_coverage <= 0.0:
            return _REASONS[0]
        if not self.has_valid_pixels:
            return _REASONS[1]
        if not self.has_effective_coverage:
            return _REASONS[0]
        return None

    @property
    def is_usable(self) -> bool:
        return self.abstention_reason is None

    @property
    def abstains(self) -> bool:
        return not self.is_usable

    @property
    def effective_coverage(self) -> np.ndarray | None:
        """Per-pixel weight, zero for pixels excluded by either mask."""
        if self.valid is None:
            return self.coverage
        mask = self.valid.astype(np.float64)
        if self.coverage is None:
            return mask
        return self.coverage * mask

    @property
    def pixels(self) -> np.ndarray:
        return self.patch.reshape(-1, 3) if self.valid is None else self.patch[self.valid]

    @property
    def weights(self) -> np.ndarray:
        if self.coverage is None:
            return np.ones(len(self.pixels), dtype=np.float64)
        valid = np.ones(self.coverage.shape, dtype=bool) if self.valid is None else self.valid
        return self.coverage[valid]

    @property
    def rgb(self) -> np.ndarray:
        """Coverage-weighted mean colour of the retained, valid pixels."""
        pixels, weights = self.pixels, self.weights
        if not len(pixels) or float(weights.sum()) <= 1e-6:
            return np.zeros(3, dtype=np.float32)
        return np.average(pixels, axis=0, weights=weights).astype(np.float32)

    @property
    def pattern(self) -> np.ndarray | None:
        """Zero-mean, unit-norm luminance over valid/covered pixels only."""
        return normalised_pattern(self.patch, self.effective_coverage)

    @property
    def descriptor(self) -> tuple[float, ...]:
        """Fixed-shape descriptor over valid/covered pixels only.

        Every pixel the effective coverage excludes is replaced by the mean of
        the retained, owned pixels and carries zero weight in the weighted
        terms, so the value an excluded pixel happens to hold - invalid or
        coverage-zero - cannot move the descriptor at all.
        """
        effective = self.effective_coverage
        patch = self.patch
        if effective is not None and bool(np.any(effective <= 0.0)):
            patch = self.patch.copy()
            patch[effective <= 0.0] = self.rgb
        return patch_descriptor(patch, effective)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FaceObservation):
            return NotImplemented
        return (
            self.identity == other.identity
            and self.normal == other.normal
            and self.support_frames == other.support_frames
            and self.provenance == other.provenance
            and self.usable_coverage == other.usable_coverage
            and self.weight == other.weight
            and np.array_equal(self.patch, other.patch)
            and np.array_equal(self.valid, other.valid)
            and np.array_equal(self.coverage, other.coverage)
        )

    def __hash__(self) -> int:
        return hash(self.identity)


def _sample_cell(sample: object) -> tuple[int, int, int]:
    """Read a sample coordinate, rejecting lossy or non-integral values."""
    try:
        raw = getattr(sample, "coordinate")
    except AttributeError as error:
        raise ValueError("sample must expose a coordinate") from error
    if isinstance(raw, (str, bytes)) or not hasattr(raw, "__iter__"):
        raise ValueError("sample coordinate must be a 3-tuple of integers")
    try:
        values = tuple(raw)
    except TypeError as error:
        raise ValueError("sample coordinate must be a 3-tuple of integers") from error
    if len(values) != 3:
        raise ValueError("sample coordinate must be a 3-tuple of integers")
    cell: list[int] = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError("sample coordinate entries must be integers, not booleans")
        if isinstance(value, (int, np.integer)):
            cell.append(int(value))
            continue
        if isinstance(value, (float, np.floating)):
            number = float(value)
            if not np.isfinite(number) or number != float(int(number)):
                raise ValueError("sample coordinate entries must be integral")
            cell.append(int(number))
            continue
        raise ValueError("sample coordinate entries must be integral numbers")
    return (cell[0], cell[1], cell[2])


def face_observation_from_sample(
    sample: object,
    *,
    frame_id: str,
    support_frames: tuple[str, ...] | None = None,
    provenance: tuple[str, ...] = (),
    weight: float = 1.0,
) -> FaceObservation:
    """Adapt a rectified face sample (duck-typed) into a retained observation.

    The sample only needs ``coordinate``, ``slot``, ``patch``, ``visible`` and
    optional ``valid``/``coverage`` attributes; the caller owns the frame
    identity and provenance. Coordinates are passed through only when they are
    exactly integral: silently truncating ``1.9`` to ``1`` would attribute the
    evidence to the wrong cell, so a lossy or non-finite value is rejected.
    """
    slot = getattr(sample, "slot", None)
    if not isinstance(slot, str) or slot not in FACE_SLOTS:
        raise ValueError(f"sample slot must be one of {FACE_SLOTS}")
    try:
        patch = getattr(sample, "patch")
    except AttributeError as error:
        raise ValueError("sample must expose a patch") from error
    try:
        visible = getattr(sample, "visible")
    except AttributeError as error:
        raise ValueError("sample must expose visible coverage") from error
    support = support_frames if support_frames is not None else (frame_id,)
    return FaceObservation(
        cell=_sample_cell(sample),
        slot=slot,
        frame_id=frame_id,
        normal=NORMAL_OF_SLOT[slot],
        patch=np.asarray(patch),
        usable_coverage=visible,  # type: ignore[arg-type]
        support_frames=tuple(support),
        valid=getattr(sample, "valid", None),
        coverage=getattr(sample, "coverage", None),
        provenance=tuple(provenance),
        weight=weight,
    )


__all__ = ["FaceObservation", "NORMAL_OF_SLOT", "face_observation_from_sample"]
