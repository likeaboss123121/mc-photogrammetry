"""Evaluation-only continuation for captured pre-filter matcher alternatives.

This module never constructs or edits a production graph record.  It binds an
arm to the complete opt-in pre-filter observation/row publication, verifies the
arm with the unchanged quadrant solver, then replays an independently built
source-bound placement universe through the public placement and volume gates.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Sequence

from mc_photogrammetry.reconstruction import plane_graph
from mc_photogrammetry.reconstruction import plane_placement
from mc_photogrammetry.reconstruction import plane_volume

SCHEMA = "mc-photogrammetry-eval-prefilter-arm-v1"


def _sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def _match_sha(match) -> str:
    return _sha(plane_graph._indexed_match_payload(match))


def _source_sha(readings, signatures) -> str:
    """Bind immutable ordered source identities without trusting an outer seal."""
    reading_rows = []
    for reading in readings:
        reading_rows.append({
            "order": list(plane_graph.reading_order_key(reading)),
            "frame": reading.frame, "axis": reading.axis,
            "alternatives": [[float(a.blocks_per_period),
                              float(a.distance_blocks), float(a.support)]
                             for a in reading.alternatives],
        })
    reading_index = {id(value): index for index, value in enumerate(readings)}
    signature_rows = []
    for signature in signatures:
        signature_rows.append({
            "reading": reading_index.get(id(signature.reading), -1),
            "b": float(signature.blocks_per_period),
            "origin": list(signature.origin_uv),
            "shape": list(signature.cells.shape),
            "cells": hashlib.sha256(signature.cells.tobytes()).hexdigest(),
            "coverage": hashlib.sha256(signature.coverage.tobytes()).hexdigest(),
        })
    return _sha({"readings": reading_rows, "signatures": signature_rows})


@dataclass(frozen=True, slots=True)
class EvalArmBudget:
    max_candidates: int
    max_rows: int
    max_active_matches: int
    max_solver_calls: int
    max_encoded_bytes: int

    def __post_init__(self):
        if any(type(getattr(self, name)) is not int or getattr(self, name) < 1
               for name in self.__dataclass_fields__):
            raise ValueError("evaluation arm budget must be positive")


@dataclass(frozen=True, slots=True)
class EvalArmRefusal:
    reason: str
    detail: tuple[tuple[str, object], ...]


@dataclass(frozen=True, slots=True)
class EvalArmPool:
    readings: tuple
    signatures: tuple
    candidates: tuple
    dispositions: tuple[str, ...]
    rows: tuple
    quadrant_constraints: tuple[tuple[str, str, int], ...]
    baseline_sha256: str
    source_sha256: str
    pool_sha256: str

    def __post_init__(self):
        if (not self.candidates or len(self.candidates) != len(self.dispositions)
                or len(self.rows) != len(self.candidates)):
            raise ValueError("pool rows are incomplete")
        if any(not isinstance(value, plane_graph.PlaneMatch)
               for value in self.candidates):
            raise ValueError("pool contains a foreign candidate")
        for index, (match, disposition, row) in enumerate(zip(
                self.candidates, self.dispositions, self.rows)):
            if (row.admitted_row_index != index or row.disposition != disposition
                    or row.match_sha256 != _match_sha(match)):
                raise ValueError("pool row does not bind its candidate")
        if (len(self.baseline_sha256) != 64 or len(self.source_sha256) != 64
                or self.source_sha256 != _source_sha(self.readings, self.signatures)):
            raise ValueError("pool source binding mismatch")
        payload = {
            "schema": SCHEMA,
            "matches": [_match_sha(value) for value in self.candidates],
            "dispositions": list(self.dispositions),
            "rows": [value.to_record() for value in self.rows],
            "constraints": [list(value) for value in self.quadrant_constraints],
            "baseline_sha256": self.baseline_sha256,
            "source_sha256": self.source_sha256,
        }
        digest = _sha(payload)
        if self.pool_sha256 not in ("", digest):
            raise ValueError("pool digest mismatch")
        object.__setattr__(self, "pool_sha256", digest)


@dataclass(frozen=True, slots=True)
class EvalArmLabel:
    admitted_rows: tuple[int, ...]
    blocks_per_reading: tuple[tuple[str, int, float], ...]
    expected_quadrants: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class EvalArmResult:
    pool_sha256: str
    active_match_sha256s: tuple[str, ...]
    placement_sha256: str
    volume_sha256: str
    coordinates: tuple[tuple[int, int, int], ...]


def build_pool(*, readings, signatures, observation, rows,
               quadrant_constraints, baseline_sha256, budget):
    try:
        if not isinstance(observation, plane_graph.SameAxisCandidateObservation):
            raise ValueError("observation")
        if not isinstance(budget, EvalArmBudget):
            raise ValueError("budget")
        if (len(observation.candidates) > budget.max_candidates
                or len(rows) > budget.max_rows):
            return EvalArmRefusal("budget_exhausted", (("stage", "pool"),))
        pool = EvalArmPool(tuple(readings), tuple(signatures),
            tuple(observation.candidates), tuple(observation.dispositions),
            tuple(rows), tuple(quadrant_constraints), baseline_sha256,
            _source_sha(readings, signatures), "")
        if len(json.dumps({"pool": pool.pool_sha256,
                           "rows": [r.to_record() for r in pool.rows]}).encode()) \
                > budget.max_encoded_bytes:
            return EvalArmRefusal("budget_exhausted", (("stage", "bytes"),))
        return pool
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        return EvalArmRefusal("provenance_mismatch",
                              (("detail", type(error).__name__),))


def continue_arm(*, pool, label, universe, supports, placement_budget,
                 volume_budget, budget):
    try:
        if not isinstance(pool, EvalArmPool) or not isinstance(label, EvalArmLabel):
            raise ValueError("arm")
        # Reconstruct to reject stale nested rows/candidates before selection.
        pool = EvalArmPool(**{name: getattr(pool, name)
                              for name in pool.__dataclass_fields__})
        if (len(label.admitted_rows) > budget.max_active_matches
                or budget.max_solver_calls < 1):
            return EvalArmRefusal("budget_exhausted", (("stage", "solver"),))
        if (not label.admitted_rows or len(set(label.admitted_rows))
                != len(label.admitted_rows)
                or any(type(index) is not int or index < 0
                       or index >= len(pool.candidates)
                       for index in label.admitted_rows)):
            raise ValueError("row selection")
        active = tuple(pool.candidates[index] for index in label.admitted_rows)
        # A feasibility arm must exercise at least one genuinely dropped row.
        if not any(pool.dispositions[index] ==
                   plane_graph.SAME_AXIS_DISPOSITION_DROPPED
                   for index in label.admitted_rows):
            raise ValueError("no dropped candidate")
        claimed_b = {(frame, index): value
                     for frame, index, value in label.blocks_per_reading}
        if len(claimed_b) != len(label.blocks_per_reading):
            raise ValueError("conflicting b")
        touched = {(m.left, m.left_plane): m.left_b for m in active}
        for match in active:
            for key, value in (((match.left, match.left_plane), match.left_b),
                               ((match.right, match.right_plane), match.right_b)):
                if key in touched and touched[key] != value:
                    raise ValueError("conflicting b")
                touched[key] = value
        if claimed_b != touched:
            raise ValueError("incomplete b")
        solved = plane_graph.solve_component(
            pool.readings, active, integer_tolerance=0.15,
            min_resultant=0.90, ambiguous_margin=0.02,
            quadrant_constraints=pool.quadrant_constraints)
        if solved.reason or tuple(sorted(solved.quadrants.items())) != \
                tuple(sorted(label.expected_quadrants)):
            return EvalArmRefusal("solver_refused", (("reason", solved.reason),))
        if dict(solved.blocks_per_period) != claimed_b:
            raise ValueError("solver b assignment mismatch")
        observed_axes = {match.left_axis for match in active}
        if observed_axes != {0, 1, 2}:
            raise ValueError("incomplete axis coverage")
        if tuple(universe.matches) != active:
            raise ValueError("active set mismatch")
        resolved = plane_placement.resolve_face_placement(
            universe=universe, supports=supports, budget=placement_budget)
        if not isinstance(resolved, plane_placement.ResolvedFacePlacement):
            return EvalArmRefusal("placement_refused",
                                  (("reason", resolved.reason),))
        volume = plane_volume.build_plane_volume(
            universe=universe, supports=supports, placement=resolved,
            placement_budget=placement_budget, budget=volume_budget)
        if not isinstance(volume, plane_volume.PlaneVolumeResult):
            return EvalArmRefusal("volume_refused", (("reason", volume.reason),))
        return EvalArmResult(pool.pool_sha256,
            tuple(_match_sha(value) for value in active),
            resolved.placement_sha256, volume.volume_sha256,
            tuple(cell.coordinate for cell in volume.cells))
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        return EvalArmRefusal("provenance_mismatch",
                              (("detail", type(error).__name__),))
