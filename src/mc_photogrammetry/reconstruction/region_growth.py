"""Image-only, edge-aware Manhattan plane regions.

This module is an experimental proposal source alongside ``boundary_candidates``.
It atomizes a bounded working image with deterministic local Lab+XY clustering,
then grows distinct axis/side hypotheses over an image graph. It accepts no
annotation, scene, camera manifest, or reference geometry.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
import hashlib
import heapq
import json
import math
import time

import numpy as np

from .calibration import (
    CalibrationResult,
    INLIER_PIXELS,
    MINIMUM_LENGTH,
    detect_segments,
    enhance,
    segment_residuals,
    structural_weights,
    vanishing_points,
    axis_directions,
)
from .boundary_candidates import (
    PlaneProposal,
    PlaneProposalResult,
    _clip_segments_to_usable_mask,
    _mask_polygon,
    _mask_iou,
)


@dataclass(frozen=True)
class RegionGrowthConfig:
    """Fixed resource and image-growth limits for the v2 proposal source."""

    analysis_max_side: int = 960
    max_input_pixels: int = 6_000_000
    atom_spacing: int = 16
    slic_iterations: int = 3
    slic_compactness: float = 10.0
    connectivity_min_component_fraction: float = 0.25
    max_atoms: int = 20_000
    max_segments: int = 5_000
    max_segment_samples: int = 128
    min_calibration_inliers: int = 12
    min_usable_fraction: float = 0.20
    min_seed_score: float = 0.26
    min_seed_margin: float = 0.035
    min_growth_score: float = 0.055
    min_seed_support: float = 12.0
    min_bridge_support: float = 18.0
    min_continuation_support: float = 5.0
    min_tangent_dominance: float = 0.84
    min_boundary_axis_alignment: float = 0.90
    max_lab_delta: float = 40.0
    max_bridge_lab_delta: float = 22.0
    hard_edge_barrier: float = 48.0
    bridge_edge_barrier: float = 15.0
    min_region_fraction: float = 0.004
    max_candidates: int = 32
    dedup_iou: float = 0.70


DEFAULT_REGION_GROWTH_CONFIG = RegionGrowthConfig()
MAX_REGION_GROWTH_PROPOSALS = 32

# Evaluation-only scalar accounting. No arrays or inference objects are retained.
_ATTRITION_MAX_ITEMS = 131072
_ATTRITION_MAX_BYTES = 16 * 1024 * 1024
_ATTRITION_EXAMPLES = 128
_ATTRITION_MAX_HANDOFFS = 1024
_attrition_state = ContextVar("scalar_attrition", default=None)
_attrition_frame = ContextVar("scalar_attrition_frame", default="")
_ATTRITION_UNSET = object()
_PERIOD_SCALAR_STAGES = ("period_measurement", "period_round", "period_profile",
                         "period_estimate", "period_reconciliation")
_LOCAL_PERIOD_SCALAR_STAGES = ("local_period_parent", "local_period_child") + tuple(
    "local_" + stage for stage in _PERIOD_SCALAR_STAGES)
_DETAILED_SCALAR_STAGES = _PERIOD_SCALAR_STAGES + _LOCAL_PERIOD_SCALAR_STAGES
_PERIOD_SCALAR_MAX_BINS = 2048
_period_scalar_state = ContextVar("period_scalar_state", default=None)


class _ScalarAttritionObserver:
    """Bounded expected/terminal identities; observer failure never fails inference."""

    def __init__(self, *, max_items=_ATTRITION_MAX_ITEMS, max_bytes=_ATTRITION_MAX_BYTES):
        self.max_items, self.max_bytes = max_items, max_bytes
        self.bytes = self.omitted = 0
        self.incomplete = False
        self.stages = {}
        self.bindings = {}
        self.checks = {}

    @staticmethod
    def _encode(value):
        def scalar(item):
            if type(item) in (list, tuple):
                return all(scalar(v) for v in item)
            if type(item) is dict:
                return all(type(k) is str and scalar(v) for k, v in item.items())
            return (item is None or type(item) in (str, int, bool)
                    or (type(item) is float and math.isfinite(item)))
        if not scalar(value):
            raise ValueError("observer accepts finite scalars only")
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode()
        if len(encoded) > 2048:
            raise ValueError("observer row exceeds scalar budget")
        return encoded

    def _stage(self, stage):
        if stage not in ("proposal", "reading", "signature", "pair", "seed", "frame") + _DETAILED_SCALAR_STAGES:
            raise ValueError("unknown observer stage")
        return self.stages.setdefault(stage, dict(
            expected={}, terminal={}, duplicates=0, unexpected=0,
            counts={}, examples=[], digest=hashlib.sha256(), omitted_examples=0))

    def _reserve(self, size):
        items = sum(len(v["expected"]) + len(v["terminal"]) + len(v["counts"])
                    for v in self.stages.values()) + len(self.bindings)
        if items >= self.max_items or self.bytes + size > self.max_bytes:
            self.omitted += 1
            self.incomplete = True
            return False
        self.bytes += size
        return True

    def expect(self, stage, identity):
        try:
            data = self._stage(stage)
            key = self._encode(identity).decode()
            if key in data["expected"]:
                data["duplicates"] += 1
            elif self._reserve(len(key) + 128):
                data["expected"][key] = identity
        except Exception:
            self.incomplete = True

    def finish(self, stage, identity, disposition, **fields):
        try:
            data = self._stage(stage)
            row = dict(identity=identity, disposition=disposition, **fields)
            encoded = self._encode(row)
            key = self._encode(identity).decode()
            if key not in data["expected"]:
                data["unexpected"] += 1
            if key in data["terminal"]:
                data["duplicates"] += 1
                return
            if not self._reserve(len(key) + 128):
                return
            data["terminal"][key] = disposition
            axis = fields.get("axis")
            group = self._encode((identity[0] if identity else "", axis, disposition,
                                 identity[4] if stage == "pair" else None,
                                 fields.get("right_axis"), fields.get("subcause"),
                                 fields.get("signal_failure_operand"),
                                 identity if stage in _DETAILED_SCALAR_STAGES else None)).decode()
            if group not in data["counts"]:
                # Full scalar details are retained per period identity, rather
                # than relying on the first 128 examples. Charge their bytes.
                detail_bytes = len(encoded) if stage in _DETAILED_SCALAR_STAGES else 0
                if not self._reserve(len(group) + 1024 + detail_bytes):
                    return
                data["counts"][group] = dict(count=0, metrics={})
                if stage in _DETAILED_SCALAR_STAGES:
                    data["counts"][group].update(identity=identity, details=row)
            aggregate = data["counts"][group]
            aggregate["count"] += 1
            for name, value in fields.items():
                if type(value) not in (int, float) or name in ("axis", "right_axis", "plane_index",
                                                             "canonical_index", "alternative_index"):
                    continue
                metric = aggregate["metrics"].setdefault(name, dict(count=0, minimum=value,
                                                                    maximum=value, total=0))
                metric.update(count=metric["count"] + 1, minimum=min(metric["minimum"], value),
                              maximum=max(metric["maximum"], value), total=metric["total"] + value)
            data["digest"].update(encoded + b"\n")
            if len(data["examples"]) < _ATTRITION_EXAMPLES:
                if self._reserve(len(encoded)):
                    data["examples"].append(row)
            else:
                data["omitted_examples"] += 1
        except Exception:
            self.incomplete = True

    def bind(self, key, identity):
        try:
            if key not in self.bindings and self._reserve(len(self._encode(identity)) + 128):
                self.bindings[key] = identity
        except Exception:
            self.incomplete = True

    def pending(self, stage, frame=None):
        data = self._stage(stage)
        return tuple(value for key, value in data["expected"].items()
                     if key not in data["terminal"] and (frame is None or value[0] == frame))

    def check(self, name, condition):
        self.checks[name] = bool(condition)

    def record(self):
        stages = {}
        for name, data in sorted(self.stages.items()):
            expected, terminal = set(data["expected"]), set(data["terminal"])
            stages[name] = dict(expected=len(expected), classified=len(terminal),
                unclassified=len(expected - terminal), unexpected=len(terminal - expected),
                multiply_classified=data["duplicates"],
                unexpected_events=data["unexpected"],
                counts=[dict(frame=json.loads(k)[0], axis=json.loads(k)[1],
                             disposition=json.loads(k)[2], right_frame=json.loads(k)[3],
                             right_axis=json.loads(k)[4], subcause=json.loads(k)[5],
                             signal_failure_operand=json.loads(k)[6], **v)
                        for k, v in sorted(data["counts"].items())],
                terminal_sha256=data["digest"].hexdigest(),
                examples=list(data["examples"]), omitted_examples=data["omitted_examples"],
                examples_status="truncated" if data["omitted_examples"] else "complete")
        closed = bool(stages) and all(not s[k] for s in stages.values()
            for k in ("unclassified", "unexpected", "multiply_classified", "unexpected_events"))
        handoffs = []
        omitted_handoffs = 0
        for key, returned in self.bindings.items():
            if key[0] != "proposal_return":
                continue
            if len(handoffs) < _ATTRITION_MAX_HANDOFFS:
                handoffs.append(dict(generated_identity=(key[1], key[2]), returned_identity=returned))
            else:
                omitted_handoffs += 1
        handoffs.sort(key=lambda row: row["generated_identity"])
        complete = closed and not self.incomplete and all(self.checks.values()) and not omitted_handoffs
        return dict(schema="mc-photogrammetry-scalar-attrition-v1",
                    status="complete" if complete else "incomplete", stages=stages,
                    reporting="complete_aggregates_bounded_examples",
                    checks=dict(sorted(self.checks.items())), omitted_accounting_events=self.omitted,
                    proposal_handoffs=handoffs, omitted_proposal_handoffs=omitted_handoffs,
                    limits=dict(accounting_items=self.max_items, accounting_bytes=self.max_bytes,
                                examples_per_stage=_ATTRITION_EXAMPLES, scalar_row_bytes=2048,
                                proposal_handoffs=_ATTRITION_MAX_HANDOFFS,
                                period_scalar_bins=_PERIOD_SCALAR_MAX_BINS,
                                period_details="charged_per_identity_full_scalar_row"))


@contextmanager
def _scalar_attrition_context(observer=_ATTRITION_UNSET, *, frame=None):
    token = _attrition_state.set(observer) if observer is not _ATTRITION_UNSET else None
    frame_token = _attrition_frame.set(frame) if frame is not None else None
    try:
        yield _attrition_state.get()
    finally:
        if frame_token is not None:
            _attrition_frame.reset(frame_token)
        if token is not None:
            _attrition_state.reset(token)


@contextmanager
def _scalar_period_context(*, identity=None, axis=None, stage_prefix=None):
    """Private optional join scope; only scalar identities/metrics are retained."""
    observer = _attrition_state.get()
    parent = _period_scalar_state.get()
    if observer is None or (identity is None and parent is None):
        yield None
        return
    state = dict(identity=identity if identity is not None else parent["identity"],
                 axis=axis if axis is not None else parent["axis"], profile=None,
                 stage_prefix=stage_prefix if stage_prefix is not None else
                 (parent["stage_prefix"] if parent is not None else ""))
    token = _period_scalar_state.set(state)
    try:
        yield state
    finally:
        _period_scalar_state.reset(token)


def _scalar_period_expect(stage, identity=None):
    observer, state = _attrition_state.get(), _period_scalar_state.get()
    if observer is not None and state is not None:
        with _scalar_observation_guard(observer):
            observer.expect(state["stage_prefix"] + stage,
                            state["identity"] if identity is None else identity)


def _scalar_period_finish(stage, reason, **fields):
    observer, state = _attrition_state.get(), _period_scalar_state.get()
    if observer is not None and state is not None:
        with _scalar_observation_guard(observer):
            observer.finish(state["stage_prefix"] + stage, state["identity"], reason,
                            axis=state["axis"], **fields)


def _scalar_period_bins(values, predicate):
    """Bounded Python reduction of an already-computed one-dimensional vector."""
    observer = _attrition_state.get()
    if len(values) > _PERIOD_SCALAR_MAX_BINS:
        observer.incomplete = True
        return None
    return sum(bool(predicate(value)) for value in values)


@contextmanager
def _scalar_observation_guard(observer):
    """Observer-only calculations must not change the inference exception path."""
    try:
        yield
    except Exception:
        observer.incomplete = True


@dataclass(frozen=True)
class _AtomGraph:
    labels: np.ndarray
    lab: np.ndarray
    usable: np.ndarray
    node_area: np.ndarray
    node_lab: np.ndarray
    node_texture: np.ndarray
    node_xy: np.ndarray
    supports: np.ndarray
    plane_scores: np.ndarray
    sides: np.ndarray
    axis_line_dirs: np.ndarray
    adjacency: tuple[tuple[tuple[int, float, float, float, float, float], ...], ...]


def _empty(reason: str, stats: dict | None = None) -> PlaneProposalResult:
    return PlaneProposalResult(status="abstain", reason=reason, proposals=(),
                               stats={} if stats is None else dict(stats))


def _as_rgb8(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
        raise ValueError("image must be a 2D or 3D numpy array")
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise ValueError("image must have 1, 3, or 4 channels")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("image dimensions must be positive")
    if not np.issubdtype(image.dtype, np.number) or not np.isfinite(image).all():
        raise ValueError("image values must be finite numbers")
    if image.dtype == np.uint8:
        values = image
    else:
        data = image.astype(np.float32)
        if data.size and float(data.max()) <= 1.0:
            data *= 255.0
        values = np.clip(data, 0.0, 255.0).astype(np.uint8)
    if values.ndim == 2:
        return np.repeat(values[:, :, None], 3, axis=2)
    if values.shape[2] == 1:
        return np.repeat(values, 3, axis=2)
    return values[:, :, :3].copy()


def _working_image(image: np.ndarray, usable_mask: np.ndarray, config: RegionGrowthConfig):
    import cv2

    rgb = _as_rgb8(image)
    height, width = rgb.shape[:2]
    usable = np.asarray(usable_mask, dtype=bool)
    if usable.shape != (height, width):
        raise ValueError("usable_mask must match the image height and width")
    if height * width > config.max_input_pixels:
        return None

    # Invalid pixels are made constant before any resize or local feature work.
    # Their supplied values can therefore never change a proposal.
    rgb[~usable] = 0
    scale = min(1.0, config.analysis_max_side / float(max(height, width)))
    work_width = max(1, int(round(width * scale)))
    work_height = max(1, int(round(height * scale)))
    if (work_width, work_height) != (width, height):
        rgb = cv2.resize(rgb, (work_width, work_height), interpolation=cv2.INTER_AREA)
        usable_small = cv2.resize(usable.astype(np.uint8), (work_width, work_height),
                                  interpolation=cv2.INTER_NEAREST) > 0
    else:
        usable_small = usable.copy()
    rgb[~usable_small] = 0
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return rgb, grey, lab, usable_small, width / work_width, height / work_height


def _local_lab_xy_atoms(
    lab: np.ndarray,
    usable: np.ndarray,
    *,
    spacing: int,
    iterations: int,
    compactness: float,
    min_component_area_fraction: float = 0.25,
    return_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, int | bool]]:
    """Overlapping deterministic SLIC assignment followed by 4-connectivity cleanup.

    Each grid seed compares within a clipped 2S-wide window against shared
    best-distance/label buffers. Seed IDs are visited in ascending order and
    strict ``<`` updates preserve the lower seed on exact ties. Work per pass is
    bounded by the sum of fixed local windows; memory is O(pixels + seeds).
    """
    height, width = usable.shape
    if spacing <= 0 or iterations < 1 or compactness < 0:
        raise ValueError("invalid local atomization parameters")
    if not 0.0 <= min_component_area_fraction <= 1.0:
        raise ValueError("connectivity component fraction must be in [0, 1]")
    rows = int(math.ceil(height / spacing))
    columns = int(math.ceil(width / spacing))
    seed_count = rows * columns
    centers_y = np.empty(seed_count, dtype=np.float32)
    centers_x = np.empty(seed_count, dtype=np.float32)
    centers_lab = np.empty((seed_count, 3), dtype=np.float32)
    seed = 0
    usable_lab_mean = (lab[usable].mean(axis=0) if usable.any()
                       else np.zeros(3, dtype=np.float32))
    for row in range(rows):
        y0, y1 = row * spacing, min(height, (row + 1) * spacing)
        for column in range(columns):
            x0, x1 = column * spacing, min(width, (column + 1) * spacing)
            patch_mask = usable[y0:y1, x0:x1]
            centers_y[seed] = min(height - 1, row * spacing + spacing / 2.0)
            centers_x[seed] = min(width - 1, column * spacing + spacing / 2.0)
            patch = lab[y0:y1, x0:x1]
            if patch_mask.any():
                centers_lab[seed] = patch[patch_mask].mean(axis=0)
            else:
                centers_lab[seed] = usable_lab_mean
            seed += 1

    spatial_weight = (compactness / float(spacing)) ** 2
    labels = np.zeros((height, width), dtype=np.int32)
    yy_all, xx_all = np.indices((height, width), dtype=np.float32)
    fallback_assignment_pixels = 0
    for _iteration in range(iterations):
        labels.fill(-1)
        best_distance = np.full((height, width), np.inf, dtype=np.float32)
        for candidate in range(seed_count):
            row, column = divmod(candidate, columns)
            anchor_y = min(height - 1, row * spacing + spacing / 2.0)
            anchor_x = min(width - 1, column * spacing + spacing / 2.0)
            y0 = max(0, int(math.floor(anchor_y - spacing)))
            y1 = min(height, int(math.ceil(anchor_y + spacing)) + 1)
            x0 = max(0, int(math.floor(anchor_x - spacing)))
            x1 = min(width, int(math.ceil(anchor_x + spacing)) + 1)
            patch = lab[y0:y1, x0:x1]
            delta = patch - centers_lab[candidate]
            color_distance = np.einsum("ijk,ijk->ij", delta, delta)
            dy = yy_all[y0:y1, x0:x1] - centers_y[candidate]
            dx = xx_all[y0:y1, x0:x1] - centers_x[candidate]
            distance = color_distance + spatial_weight * (dy * dy + dx * dx)
            current = best_distance[y0:y1, x0:x1]
            choose = usable[y0:y1, x0:x1] & (distance < current)
            current[choose] = distance[choose]
            labels[y0:y1, x0:x1][choose] = candidate

        # The fixed grid-centered 2S windows cover the full image. This bounded
        # fallback protects mask authority if dimensions/rounding ever violate
        # that proof without consulting any excluded pixels.
        uncovered = usable & (labels < 0)
        if uncovered.any():
            fallback_assignment_pixels += int(uncovered.sum())
            yy, xx = np.nonzero(uncovered)
            rows_for_pixel = np.minimum(yy // spacing, rows - 1)
            columns_for_pixel = np.minimum(xx // spacing, columns - 1)
            labels[yy, xx] = rows_for_pixel * columns + columns_for_pixel

        valid = usable & (labels >= 0)
        flat_ids = labels[valid]
        counts = np.bincount(flat_ids, minlength=seed_count).astype(np.float32)
        if not np.any(counts):
            break
        lab_sums = np.stack([
            np.bincount(flat_ids, weights=lab[:, :, channel][valid],
                        minlength=seed_count)
            for channel in range(3)
        ], axis=1).astype(np.float32)
        y_sums = np.bincount(flat_ids, weights=yy_all[valid], minlength=seed_count)
        x_sums = np.bincount(flat_ids, weights=xx_all[valid], minlength=seed_count)
        nonempty = counts > 0
        centers_lab[nonempty] = lab_sums[nonempty] / counts[nonempty, None]
        centers_y[nonempty] = y_sums[nonempty] / counts[nonempty]
        centers_x[nonempty] = x_sums[nonempty] / counts[nonempty]

    labels[~usable] = -1
    components = _split_disconnected_atoms(labels)
    pre_cleanup_count = int(components.max()) + 1 if components.size and components.max() >= 0 else 0
    min_component_area = max(1, int(math.ceil(
        spacing * spacing * min_component_area_fraction)))
    cleaned, merge_count, isolated_small_count = _merge_small_components(
        components, min_component_area)
    post_cleanup_count = (int(cleaned.max()) + 1
                          if cleaned.size and cleaned.max() >= 0 else 0)
    mask_preserved = bool(np.array_equal(cleaned >= 0, usable))
    atom_stats: dict[str, int | bool] = {
        "seed_count": seed_count,
        "components_before_cleanup": pre_cleanup_count,
        "connectivity_min_component_area": min_component_area,
        "connectivity_cleanup_merges": merge_count,
        "isolated_small_components_retained": isolated_small_count,
        "components_after_cleanup": post_cleanup_count,
        "connectivity_fallback_assignment_pixels": fallback_assignment_pixels,
        "connectivity_mask_preserved": mask_preserved,
    }
    if return_stats:
        return cleaned, atom_stats
    return cleaned


def _split_disconnected_atoms(labels: np.ndarray) -> np.ndarray:
    """Ensure one SLIC label cannot connect distant lookalike image patches."""
    import cv2

    valid_positions = np.flatnonzero(labels.ravel() >= 0)
    if not valid_positions.size:
        return np.full(labels.shape, -1, dtype=np.int32)
    flat = labels.ravel()
    order = np.argsort(flat[valid_positions], kind="stable")
    positions = valid_positions[order]
    ordered_ids = flat[positions]
    starts = np.r_[0, np.flatnonzero(ordered_ids[1:] != ordered_ids[:-1]) + 1,
                   len(ordered_ids)]
    height, width = labels.shape
    compact = np.full(labels.shape, -1, dtype=np.int32)
    next_id = 0
    for begin, end in zip(starts[:-1], starts[1:]):
        coords = positions[begin:end]
        ys, xs = np.divmod(coords, width)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        seed_id = int(ordered_ids[begin])
        crop = (labels[y0:y1, x0:x1] == seed_id).astype(np.uint8)
        component_count, components = cv2.connectedComponents(crop, connectivity=4)
        for component in range(1, component_count):
            component_mask = components == component
            if not np.any(component_mask):
                continue
            view = compact[y0:y1, x0:x1]
            view[component_mask] = next_id
            next_id += 1
    return compact


def _merge_small_components(
    labels: np.ndarray, minimum_area: int,
) -> tuple[np.ndarray, int, int]:
    """Merge small 4-connected atoms by shared boundary; retain isolated islands.

    The component adjacency graph has at most two undirected edges per image
    pixel. A heap visits small components in (area, stable component ID) order.
    Every merge follows a positive shared boundary, so each resulting ID remains
    4-connected. Large components are never merged merely to meet a cap.
    """
    if minimum_area < 1:
        raise ValueError("minimum component area must be positive")
    height, width = labels.shape
    valid = labels >= 0
    ids = labels[valid]
    component_count = int(ids.max()) + 1 if ids.size else 0
    if component_count == 0:
        return np.full(labels.shape, -1, dtype=np.int32), 0, 0
    areas = np.bincount(ids, minlength=component_count).astype(np.int64)

    left_a, left_b = labels[:, :-1], labels[:, 1:]
    up_a, up_b = labels[:-1, :], labels[1:, :]
    horizontal = (left_a >= 0) & (left_b >= 0) & (left_a != left_b)
    vertical = (up_a >= 0) & (up_b >= 0) & (up_a != up_b)
    edge_a = np.concatenate((left_a[horizontal], up_a[vertical])).astype(np.int64)
    edge_b = np.concatenate((left_b[horizontal], up_b[vertical])).astype(np.int64)
    if edge_a.size:
        low = np.minimum(edge_a, edge_b)
        high = np.maximum(edge_a, edge_b)
        keys, shared_counts = np.unique(low * component_count + high,
                                        return_counts=True)
    else:
        keys = np.empty(0, dtype=np.int64)
        shared_counts = np.empty(0, dtype=np.int64)
    adjacency: list[dict[int, int]] = [{} for _ in range(component_count)]
    for key, shared in zip(keys, shared_counts):
        first = int(key // component_count)
        second = int(key % component_count)
        boundary = int(shared)
        adjacency[first][second] = boundary
        adjacency[second][first] = boundary

    parent = np.arange(component_count, dtype=np.int32)
    heap = [(int(areas[node]), node) for node in range(component_count)
            if areas[node] < minimum_area]
    heapq.heapify(heap)
    merges = 0
    while heap:
        queued_area, source = heapq.heappop(heap)
        if parent[source] != source or int(areas[source]) != queued_area:
            continue
        if queued_area >= minimum_area:
            continue
        neighbors = adjacency[source]
        if not neighbors:
            continue
        target = min(neighbors, key=lambda node: (-neighbors[node], node))
        for neighbor, boundary in tuple(neighbors.items()):
            if neighbor == target:
                continue
            combined_boundary = adjacency[target].get(neighbor, 0) + boundary
            adjacency[target][neighbor] = combined_boundary
            adjacency[neighbor].pop(source, None)
            adjacency[neighbor][target] = combined_boundary
        adjacency[target].pop(source, None)
        adjacency[source].clear()
        parent[source] = target
        areas[target] += areas[source]
        areas[source] = 0
        merges += 1
        if areas[target] < minimum_area:
            heapq.heappush(heap, (int(areas[target]), target))

    roots = np.arange(component_count, dtype=np.int32)
    for node in range(component_count):
        root = node
        while parent[root] != root:
            parent[root] = parent[parent[root]]
            root = int(parent[root])
        roots[node] = root
    active_roots = np.unique(roots)
    root_to_compact = np.full(component_count, -1, dtype=np.int32)
    root_to_compact[active_roots] = np.arange(len(active_roots), dtype=np.int32)
    remap = root_to_compact[roots]
    cleaned = labels.copy()
    cleaned[valid] = remap[labels[valid]]
    isolated_small = sum(
        int(areas[node] < minimum_area and not adjacency[node])
        for node in range(component_count)
        if parent[node] == node and node < len(adjacency))
    return cleaned, merges, isolated_small


def _node_features(labels: np.ndarray, lab: np.ndarray):
    valid = labels >= 0
    ids = labels[valid]
    count = int(ids.max()) + 1 if ids.size else 0
    area = np.bincount(ids, minlength=count).astype(np.int32)
    yy, xx = np.indices(labels.shape, dtype=np.float32)
    node_lab = np.stack([
        np.bincount(ids, weights=lab[:, :, channel][valid], minlength=count)
        for channel in range(3)
    ], axis=1).astype(np.float32)
    node_square = np.stack([
        np.bincount(ids, weights=(lab[:, :, channel][valid] ** 2), minlength=count)
        for channel in range(3)
    ], axis=1).astype(np.float32)
    node_y = np.bincount(ids, weights=yy[valid], minlength=count)
    node_x = np.bincount(ids, weights=xx[valid], minlength=count)
    nonempty = np.maximum(area, 1)
    node_lab /= nonempty[:, None]
    node_square /= nonempty[:, None]
    node_texture = np.sqrt(np.maximum(node_square[:, 0] - node_lab[:, 0] ** 2, 0.0))
    xy = np.stack([node_x / nonempty, node_y / nonempty], axis=1).astype(np.float32)
    return area, node_lab, node_texture.astype(np.float32), xy


def _graph_adjacency(labels: np.ndarray, lab: np.ndarray, node_lab: np.ndarray,
                     node_texture: np.ndarray):
    height, width = labels.shape
    node_count = len(node_lab)
    pairs_all = []
    grads_all = []
    orientation_all = []
    contact_count_all = []
    for orientation, (first, second, pixel_delta) in enumerate((
        (labels[:, :-1], labels[:, 1:], lab[:, :-1] - lab[:, 1:]),
        (labels[:-1, :], labels[1:, :], lab[:-1, :] - lab[1:, :]),
    )):
        touching = (first >= 0) & (second >= 0) & (first != second)
        if not np.any(touching):
            continue
        a = first[touching].astype(np.int64)
        b = second[touching].astype(np.int64)
        low, high = np.minimum(a, b), np.maximum(a, b)
        keys = low * node_count + high
        gradients = np.linalg.norm(pixel_delta[touching], axis=1)
        unique_keys, inverse = np.unique(keys, return_inverse=True)
        sums = np.bincount(inverse, weights=gradients)
        counts = np.bincount(inverse)
        pairs_all.append(np.stack([unique_keys // node_count,
                                   unique_keys % node_count], axis=1))
        grads_all.append(sums / np.maximum(counts, 1))
        orientation_all.append(np.full(len(unique_keys), orientation, dtype=np.int8))
        contact_count_all.append(counts.astype(np.float32))
    if not pairs_all:
        return tuple(() for _ in range(node_count))
    pairs = np.concatenate(pairs_all, axis=0)
    grads = np.concatenate(grads_all).astype(np.float32)
    orientations = np.concatenate(orientation_all)
    contacts = np.concatenate(contact_count_all)
    unique_pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)
    edge_gradient = np.bincount(inverse, weights=grads) / np.maximum(
        np.bincount(inverse), 1)
    vertical_contacts = np.bincount(
        inverse, weights=contacts * (orientations == 0))
    horizontal_contacts = np.bincount(
        inverse, weights=contacts * (orientations == 1))
    # A left/right pixel contact has a vertical boundary tangent; a
    # top/bottom contact has a horizontal boundary tangent.
    boundary_vertical = vertical_contacts >= horizontal_contacts
    adjacency = [[] for _ in range(node_count)]
    for pair_index, ((a, b), gradient) in enumerate(zip(unique_pairs, edge_gradient)):
        color_delta = float(np.linalg.norm(node_lab[a] - node_lab[b]))
        texture_delta = float(abs(node_texture[a] - node_texture[b]))
        tangent_x, tangent_y = ((0.0, 1.0) if boundary_vertical[pair_index]
                                 else (1.0, 0.0))
        adjacency[int(a)].append((int(b), float(gradient), color_delta, texture_delta,
                                  tangent_x, tangent_y))
        adjacency[int(b)].append((int(a), float(gradient), color_delta, texture_delta,
                                  tangent_x, tangent_y))
    return tuple(tuple(sorted(edges, key=lambda item: item[0])) for edges in adjacency)


_RASTER_MAX_SIDE = 960
_RASTER_MAX_SEGMENTS = 5000
_RASTER_MAX_VISITS = 4_800_000
_RASTER_MAX_NODES = 20_000
_RASTER_SCRATCH_BYTES = 256 * 1024
_RASTER_INVALID = "raster_support_invalid_input"
_RASTER_BUDGET = "raster_support_budget_exceeded"


class _RasterSupportWorkspace:
    """One reused segment: seven int64 and three boolean vectors of length 960.

    Live numeric storage is 56,640 bytes. Advanced-index gathers add at most
    8,640 bytes (one integer label and one mask vector); row arithmetic adds
    at most 96 bytes. In-place heapsort has constant auxiliary workspace.
    Including 4096 bytes each for headers/views and scalar/call state keeps
    helper scratch below 74 KiB, well inside the private 256 KiB limit.
    Frame labels/mask/segments/associations and returned support are separate
    retained inputs/output, never copied into per-segment scratch.
    """

    def __init__(self):
        self.steps = np.arange(_RASTER_MAX_SIDE, dtype=np.int64)
        self.numerator = np.empty(_RASTER_MAX_SIDE, dtype=np.int64)
        self.quotient = np.empty(_RASTER_MAX_SIDE, dtype=np.int64)
        self.remainder = np.empty(_RASTER_MAX_SIDE, dtype=np.int64)
        self.x = np.empty(_RASTER_MAX_SIDE, dtype=np.int64)
        self.y = np.empty(_RASTER_MAX_SIDE, dtype=np.int64)
        self.ids = np.empty(_RASTER_MAX_SIDE, dtype=np.int64)
        self.round_up = np.empty(_RASTER_MAX_SIDE, dtype=bool)
        self.valid = np.empty(_RASTER_MAX_SIDE, dtype=bool)
        self.other = np.empty(_RASTER_MAX_SIDE, dtype=bool)

    @property
    def numeric_bytes(self):
        return sum(value.nbytes for value in vars(self).values())


def _raster_endpoints(segment, width, height):
    """Round finite endpoints to even; canonicalize without moving a pixel."""
    if len(segment) != 4:
        raise ValueError(_RASTER_INVALID)
    endpoints = []
    for coordinate in segment:
        try:
            value = float(coordinate)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(_RASTER_INVALID) from None
        if not math.isfinite(value):
            raise ValueError(_RASTER_INVALID)
        endpoints.append(round(value))
    x0, y0, x1, y1 = endpoints
    if not (0 <= x0 < width and 0 <= x1 < width
            and 0 <= y0 < height and 0 <= y1 < height):
        raise ValueError(_RASTER_INVALID)
    if (x0, y0) > (x1, y1):
        x0, y0, x1, y1 = x1, y1, x0, y0
    count = max(abs(x1 - x0), abs(y1 - y0)) + 1
    if count > _RASTER_MAX_SIDE:
        raise ValueError(_RASTER_BUDGET)
    return x0, y0, x1, y1, count


def _thin_centerline(segment, width, height, workspace):
    """Return workspace views of unique nearest-even integer centerline pixels.

    Minor coordinates use exact rational divmod, including negative slopes;
    no floating interpolation, supercover, or universal reflection-at-ties
    claim. Returned views are overwritten by the next segment.
    """
    x0, y0, x1, y1, count = _raster_endpoints(segment, width, height)
    xs, ys = workspace.x[:count], workspace.y[:count]
    if count == 1:
        xs[0], ys[0] = x0, y0
        return xs, ys
    dx, dy = x1 - x0, y1 - y0
    if abs(dx) >= abs(dy):
        dominant, minor, start, sign, origin, delta = xs, ys, x0, 1, y0, dy
    else:
        dominant, minor, start = ys, xs, y0
        sign, origin, delta = (1 if dy > 0 else -1), x0, dx
    denominator = count - 1
    steps = workspace.steps[:count]
    numerator, quotient = workspace.numerator[:count], workspace.quotient[:count]
    remainder = workspace.remainder[:count]
    rounded, tied = workspace.round_up[:count], workspace.valid[:count]
    np.multiply(steps, sign, out=dominant)
    np.add(dominant, start, out=dominant)
    np.multiply(steps, delta, out=numerator)
    np.add(numerator, origin * denominator, out=numerator)
    np.floor_divide(numerator, denominator, out=quotient)
    np.remainder(numerator, denominator, out=remainder)
    np.multiply(remainder, 2, out=numerator)
    np.greater(numerator, denominator, out=rounded)
    np.equal(numerator, denominator, out=tied)
    np.bitwise_and(quotient, 1, out=remainder)
    np.logical_and(tied, remainder, out=tied)
    np.logical_or(rounded, tied, out=rounded)
    np.add(quotient, rounded, out=minor)
    return xs, ys


def _raster_validate_frame(grey, usable, labels, config):
    # Only literal True enters here, before detection or residual buffers.
    if (type(config.max_segments) is not int
            or not 0 <= config.max_segments <= _RASTER_MAX_SEGMENTS):
        raise ValueError(_RASTER_BUDGET)
    if (not isinstance(grey, np.ndarray) or grey.ndim != 2
            or not isinstance(labels, np.ndarray)
            or labels.shape != grey.shape or labels.dtype.kind not in "iu"
            or not isinstance(usable, np.ndarray)
            or usable.shape != grey.shape or usable.dtype != np.dtype(bool)):
        raise ValueError(_RASTER_INVALID)
    if not all(0 < dimension <= _RASTER_MAX_SIDE for dimension in grey.shape):
        raise ValueError(_RASTER_BUDGET)
    maximum = int(labels.max())
    node_count = maximum + 1 if maximum >= 0 else 0
    if node_count > _RASTER_MAX_NODES:
        raise ValueError(_RASTER_BUDGET)
    return node_count


def _raster_validate_detected(segments):
    if (not isinstance(segments, np.ndarray) or segments.ndim != 2
            or segments.shape[1] != 4 or segments.dtype.kind not in "iuf"):
        raise ValueError(_RASTER_INVALID)
    for segment in segments:
        if any(not math.isfinite(float(value)) for value in segment):
            raise ValueError(_RASTER_INVALID)


def _raster_validate_segments(segments, width, height):
    _raster_validate_detected(segments)
    total = 0
    for segment in segments:
        total += _raster_endpoints(segment, width, height)[-1]
        if total > _RASTER_MAX_VISITS:
            raise ValueError(_RASTER_BUDGET)
    return total


def _raster_validate_evidence(lengths, weights, associations, count):
    if (np.shape(lengths) != (count,) or np.shape(weights) != (count,)
            or np.shape(associations) != (count, 3)):
        raise ValueError(_RASTER_INVALID)
    # Scalar checks avoid unchecked frame-sized boolean temporaries.
    for length, weight, row in zip(lengths, weights, associations):
        if (not math.isfinite(float(length)) or length < 0
                or not math.isfinite(float(weight)) or weight < 0
                or any(not math.isfinite(float(value)) or value < 0 for value in row)):
            raise ValueError(_RASTER_INVALID)


def _raster_allocate_segment(support, labels, usable, segment, length,
                             weight, association, workspace):
    """Credit exact usable labels; invalid share remains unsupported."""
    height, width = labels.shape
    xs, ys = _thin_centerline(segment, width, height, workspace)
    count = len(xs)
    ids = workspace.ids[:count]
    valid, other = workspace.valid[:count], workspace.other[:count]
    # Both gathers are <=960 entries; no flattened copy of frame inputs.
    np.copyto(ids, labels[ys, xs], casting="unsafe")
    np.greater_equal(ids, 0, out=valid)
    np.less(ids, len(support), out=other)
    np.logical_and(valid, other, out=valid)
    np.logical_and(valid, usable[ys, xs], out=valid)
    np.logical_not(valid, out=other)
    np.copyto(ids, -1, where=other)
    ids.sort(kind="heapsort")
    # Group only visited labels, with no max_atoms histogram/dict/unique temp.
    start = 0
    while start < count:
        node = int(ids[start])
        end = start + 1
        while end < count and ids[end] == node:
            end += 1
        if node >= 0:
            contribution = float(length) * float(weight) * (end - start) / count
            support[node] += (contribution * association).astype(np.float32)
        start = end


def _paired_support_observe(observer, method, *args, **kwargs):
    """An optional diagnostic sink cannot replace inference with telemetry."""
    if observer is None:
        return
    try:
        getattr(observer, method)(*args, **kwargs)
    except Exception:
        try:
            observer.discard("observer_error")
        except Exception:
            pass


def _validate_support_line_weight_policy(policy: str) -> str:
    """Validate a nonserialized internal experiment, never ordinary settings."""
    if type(policy) is not str or policy not in ("structural", "unit"):
        raise ValueError("invalid internal support line-weight policy")
    return policy


def _segment_support(
    grey: np.ndarray,
    usable: np.ndarray,
    labels: np.ndarray,
    calibration: CalibrationResult,
    config: RegionGrowthConfig,
    *,
    original_pixel_scale: tuple[float, float] = (1.0, 1.0),
    _raster_support=False,
    _support_observer=None,
    _support_line_weight_policy: str = "structural",
 ) -> tuple[np.ndarray, dict[str, int | float]]:
    weight_policy = _validate_support_line_weight_policy(_support_line_weight_policy)
    raster = _raster_support is True
    if raster and _support_observer is not None:
        _paired_support_observe(_support_observer, "discard", "raster_authority")
        _support_observer = None
    if raster:
        _raster_validate_frame(grey, usable, labels, config)
    height, width = grey.shape
    if original_pixel_scale == (1.0, 1.0):
        raw = detect_segments(enhance(grey))
        if raster:
            _raster_validate_detected(raw)
    else:
        # Defer length admission until coordinates are measured in original
        # pixels. Rounded resize dimensions can give different x/y scales.
        raw = detect_segments(enhance(grey), minimum_length=0.0)
        if raster:
            _raster_validate_detected(raw)
        coordinates = np.asarray(raw, dtype=np.float64)
        scale_x, scale_y = original_pixel_scale
        original_lengths = np.hypot(
            (coordinates[:, 2] - coordinates[:, 0]) * scale_x,
            (coordinates[:, 3] - coordinates[:, 1]) * scale_y,
        )
        raw = raw[original_lengths >= MINIMUM_LENGTH]
    diagnostics: dict[str, int | float] = {
        "segments_detected": int(len(raw)),
        "segments_clipped": 0,
        "segments_with_axis_assignment": 0,
        "atoms_with_support": 0,
        "support_mass": 0.0,
    }
    node_count = int(labels.max()) + 1 if labels.size and labels.max() >= 0 else 0
    def empty_return(support, segments):
        if _support_observer is not None:
            _paired_support_observe(_support_observer, "begin", grey, usable, labels,
                calibration, config, segments, None, None, None, node_count,
                original_pixel_scale)
            _paired_support_observe(_support_observer, "finalize", support)
        return support, diagnostics
    if not len(raw):
        return empty_return(np.zeros((node_count, 3), dtype=np.float32 if raster else float), raw)
    segments = _clip_segments_to_usable_mask(raw, usable)
    diagnostics["segments_clipped"] = int(len(segments))
    if not len(segments):
        return empty_return(np.zeros((node_count, 3), dtype=np.float32 if raster else float), segments)
    if raster:
        _raster_validate_detected(segments)
    reverse = ((segments[:, 0] > segments[:, 2])
               | ((segments[:, 0] == segments[:, 2])
                  & (segments[:, 1] > segments[:, 3])))
    segments[reverse] = segments[reverse][:, [2, 3, 0, 1]]
    order = np.lexsort((segments[:, 3], segments[:, 2], segments[:, 1], segments[:, 0]))
    segments = segments[order]
    lengths = np.hypot(segments[:, 2] - segments[:, 0],
                       segments[:, 3] - segments[:, 1])
    if len(segments) > config.max_segments:
        keep = np.argsort(-lengths, kind="stable")[:config.max_segments]
        segments = segments[keep]
        lengths = lengths[keep]
        order = np.lexsort((segments[:, 3], segments[:, 2], segments[:, 1], segments[:, 0]))
        segments = segments[order]
        lengths = lengths[order]

    if raster:
        _raster_validate_segments(segments, width, height)
    support = None if raster else np.zeros((node_count, 3), dtype=np.float32)
    if not len(segments):
        return empty_return(np.zeros((node_count, 3), dtype=np.float32) if raster else support,
                            segments)
    points = vanishing_points(
        calibration.focal_pixels, calibration.yaw_degrees,
        calibration.pitch_degrees, width, height, calibration.roll_degrees,
        calibration.principal_point,
    )
    residuals = segment_residuals(segments, points)
    if raster:
        if np.shape(residuals) != (len(segments), 3):
            raise ValueError(_RASTER_INVALID)
        for row in residuals:
            if any(not math.isfinite(float(value)) for value in row):
                raise ValueError(_RASTER_INVALID)
    softness = max(INLIER_PIXELS * 1.35, 1.0)
    associations = np.exp(-0.5 * (residuals / softness) ** 2)
    associations[residuals > 3.0 * INLIER_PIXELS] = 0.0
    association_sums = associations.sum(axis=1)
    valid_lines = association_sums > 1e-9
    diagnostics["segments_with_axis_assignment"] = int(np.count_nonzero(valid_lines))
    associations[valid_lines] /= association_sums[valid_lines, None]
    associations[~valid_lines] = 0.0
    line_weights = (np.ones(len(segments)) if weight_policy == "unit"
                    else structural_weights(grey, segments))
    if raster:
        _raster_validate_evidence(lengths, line_weights, associations, len(segments))
        support = np.zeros((node_count, 3), dtype=np.float32)
        workspace = _RasterSupportWorkspace()
    if _support_observer is not None:
        _paired_support_observe(_support_observer, "begin", grey, usable, labels,
            calibration, config, segments, lengths, line_weights, associations,
            node_count, original_pixel_scale)
    step = max(float(config.atom_spacing) / 2.0, 4.0)
    for index, segment in enumerate(segments):
        if not valid_lines[index]:
            if _support_observer is not None:
                _paired_support_observe(_support_observer, "segment", index,
                    segments, lengths, line_weights, associations, None, None)
            continue
        if raster:
            _raster_allocate_segment(support, labels, usable, segment,
                                     lengths[index], line_weights[index],
                                     associations[index], workspace)
            continue
        x0, y0, x1, y1 = segment
        length = float(lengths[index])
        sample_count = min(config.max_segment_samples,
                           max(2, int(math.ceil(length / step)) + 1))
        fractions = np.linspace(0.0, 1.0, sample_count)
        xs = np.clip(np.rint(x0 + fractions * (x1 - x0)).astype(int), 0, width - 1)
        ys = np.clip(np.rint(y0 + fractions * (y1 - y0)).astype(int), 0, height - 1)
        ids = labels[ys, xs]
        for node in np.unique(ids[ids >= 0]):
            count = int(np.count_nonzero(ids == node))
            contribution = (length * float(line_weights[index]) * count
                            / float(sample_count))
            support[int(node)] += (contribution * associations[index]).astype(np.float32)
        if _support_observer is not None:
            _paired_support_observe(_support_observer, "segment", index,
                segments, lengths, line_weights, associations, ids, sample_count)
    diagnostics["atoms_with_support"] = int(np.count_nonzero(support.sum(axis=1) > 1e-6))
    diagnostics["support_mass"] = float(support.sum())
    if _support_observer is not None:
        _paired_support_observe(_support_observer, "finalize", support)
    return support, diagnostics


def _normal_scores(supports: np.ndarray) -> np.ndarray:
    scores = np.zeros_like(supports, dtype=np.float32)
    for axis in range(3):
        a, b = [other for other in range(3) if other != axis]
        sa, sb, sc = supports[:, axis], supports[:, a], supports[:, b]
        in_plane = sb + sc
        total = in_plane + sa
        balanced = 2.0 * np.sqrt(sb * sc) / np.maximum(in_plane, 1e-6)
        absent = in_plane / np.maximum(total, 1e-6)
        reliability = in_plane / (in_plane + 24.0)
        scores[:, axis] = np.clip(balanced * absent * reliability, 0.0, 1.0)
    return scores


def _atom_sides(node_xy: np.ndarray, calibration: CalibrationResult,
                width: int, height: int) -> np.ndarray:
    from .calibration import axis_directions

    centre_x, centre_y = calibration.principal_point or (width / 2.0, height / 2.0)
    rays = np.stack([
        (node_xy[:, 0] - centre_x) / calibration.focal_pixels,
        (centre_y - node_xy[:, 1]) / calibration.focal_pixels,
        np.ones(len(node_xy), dtype=np.float32),
    ], axis=1)
    directions = axis_directions(calibration.yaw_degrees, calibration.pitch_degrees,
                                 calibration.roll_degrees)
    signs = np.sign(rays @ directions.T).astype(np.int8)
    return signs


def _axis_line_directions(node_xy: np.ndarray, calibration: CalibrationResult,
                          width: int, height: int) -> np.ndarray:
    """Return each Manhattan axis line direction at every atom centroid."""
    points = vanishing_points(
        calibration.focal_pixels, calibration.yaw_degrees,
        calibration.pitch_degrees, width, height, calibration.roll_degrees,
        calibration.principal_point,
    )
    result = np.empty((len(node_xy), 3, 2), dtype=np.float32)
    for axis, point in enumerate(points):
        if abs(float(point[2])) > 1e-9:
            direction = point[:2] / point[2] - node_xy
        else:
            direction = np.broadcast_to(point[:2], (len(node_xy), 2)).copy()
        lengths = np.linalg.norm(direction, axis=1)
        invalid = lengths < 1e-6
        if np.any(invalid):
            direction[invalid] = (1.0, 0.0)
            lengths[invalid] = 1.0
        result[:, axis] = direction / lengths[:, None]
    return result


def _diagnostic_atom_membership(labels: np.ndarray, points: np.ndarray) -> tuple[int, ...]:
    """Bounded transient source-label lookup; no production policy or ownership.

    Coordinates must already be in this atomization's working frame. No
    implicit resize, clipping or original/working coordinate conversion occurs.
    Invalid/out-of-frame points receive -1, never a border atom assignment.
    """
    if (not isinstance(labels, np.ndarray) or labels.ndim != 2
            or not np.issubdtype(labels.dtype, np.integer)
            or min(labels.shape) < 1 or max(labels.shape) > 960):
        raise ValueError("invalid diagnostic atom labels")
    if (not isinstance(points, np.ndarray) or points.ndim != 2
            or points.shape[1] != 2 or len(points) > 64
            or not np.issubdtype(points.dtype, np.number)
            or np.issubdtype(points.dtype, np.complexfloating)):
        raise ValueError("invalid diagnostic membership points")
    height, width = labels.shape
    result = []
    for x, y in points:
        if (not math.isfinite(float(x)) or not math.isfinite(float(y))
                or not 0 <= x <= width - 1 or not 0 <= y <= height - 1):
            result.append(-1)
            continue
        node = int(labels[int(np.rint(y)), int(np.rint(x))])
        if node < -1 or node >= 20_000:
            raise ValueError("invalid diagnostic atom id")
        result.append(node)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class _DiagnosticAtomSnapshot:
    """Immutable source-produced atom census, private and never serialized."""
    frame_id: str
    session_sha256: str
    pixel_sha256: str
    source_shape: tuple[int, ...]
    working_shape: tuple[int, int]
    native_to_working_scale: tuple[float, float]
    coordinate_rule: str
    labels_bytes: bytes
    usable_bytes: bytes
    labels_sha256: str
    usable_sha256: str
    native_usable_sha256: str
    native_usable_bytes: bytes
    snapshot_sha256: str


def _diagnostic_snapshot_digest(snapshot):
    payload = {name: getattr(snapshot, name) for name in snapshot.__dataclass_fields__
               if name not in ("labels_bytes", "usable_bytes", "native_usable_bytes", "snapshot_sha256")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"))
                          .encode("ascii")).hexdigest()


def _diagnostic_native_mask_digest(frame_id, source_shape, mask_bytes):
    header = json.dumps([frame_id, source_shape[:2]], separators=(",", ":")).encode("ascii")
    return hashlib.sha256(b"mc-native-gui-usable-v1\0" + header + b"\0" + mask_bytes).hexdigest()


def _diagnostic_native_to_working(points, scales):
    """OpenCV resize pixel-center convention, with NumPy's nearest/tie rule.

    No clipping: valid native-edge centers outside the working-center range
    conservatively produce unsupported membership at the subsequent lookup.
    """
    if (not isinstance(points, np.ndarray) or points.ndim != 2
            or points.shape[1] != 2 or len(points) > 64
            or type(scales) is not tuple or len(scales) != 2
            or any(type(scale) is not float or not math.isfinite(scale) or scale <= 0
                   for scale in scales)):
        raise ValueError("invalid diagnostic center mapping")
    return (points.astype(np.float64) + .5) / np.asarray(scales) - .5


def _diagnostic_atom_snapshot(image, native_usable, labels, working_usable, scales, binding):
    """Capture exact same-pass inputs/labels; <=10,608,000 retained census bytes.

    Native mask and image hashes are computed from the actual proposal inputs;
    only mask/census bytes survive. No labels or adjacency are supplied
    by the diagnostic consumer. Coordinates use the proposal's axis scales.
    """
    from mc_photogrammetry.pipeline.image_derived_session import image_sha256
    if (type(binding) is not tuple or len(binding) != 3 or not binding[0]
            or any(type(v) is not str for v in binding)
            or any(len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
                   for v in binding[1:])):
        raise ValueError("invalid diagnostic frame binding")
    if image_sha256(image) != binding[2]:
        raise ValueError("diagnostic source pixel mismatch")
    if (image.shape[0]*image.shape[1] > 6_000_000 or labels.ndim != 2
            or max(labels.shape) > 960 or min(labels.shape) < 1
            or labels.shape != working_usable.shape or native_usable.shape != image.shape[:2]
            or labels.dtype != np.int32 or working_usable.dtype != bool
            or native_usable.dtype != bool or not np.array_equal(labels >= 0, working_usable)
            or np.any(labels < -1) or np.any(labels >= 20_000)):
        raise ValueError("invalid diagnostic census")
    expected = (image.shape[1]/labels.shape[1], image.shape[0]/labels.shape[0])
    if type(scales) is not tuple or scales != expected:
        raise ValueError("diagnostic axis scale mismatch")
    label_bytes, mask_bytes = labels.astype("<i4", copy=False).tobytes(), working_usable.tobytes()
    snapshot = _DiagnosticAtomSnapshot(
        *binding, tuple(image.shape), tuple(labels.shape), scales,
        "native_pixel_centers_plus_half_divided_by_axis_scale_minus_half_numpy_rint_abstain_edges",
        label_bytes, mask_bytes, hashlib.sha256(label_bytes).hexdigest(),
        hashlib.sha256(mask_bytes).hexdigest(),
        _diagnostic_native_mask_digest(binding[0], tuple(image.shape), native_usable.tobytes()),
        native_usable.tobytes(), "")
    return replace(snapshot, snapshot_sha256=_diagnostic_snapshot_digest(snapshot))


def _build_graph(
    labels: np.ndarray,
    grey: np.ndarray,
    lab: np.ndarray,
    usable: np.ndarray,
    calibration: CalibrationResult,
    config: RegionGrowthConfig,
    *,
    original_pixel_scale: tuple[float, float] = (1.0, 1.0),
    _support_line_weight_policy: str = "structural",
) -> tuple[_AtomGraph | None, dict[str, int | float]]:
    weight_policy = _validate_support_line_weight_policy(_support_line_weight_policy)
    node_area, node_lab, node_texture, node_xy = _node_features(labels, lab)
    if len(node_area) == 0:
        return None, {}
    adjacency = _graph_adjacency(labels, lab, node_lab, node_texture)
    supports, evidence = _segment_support(
        grey, usable, labels, calibration, config,
        original_pixel_scale=original_pixel_scale,
        _support_line_weight_policy=weight_policy)
    scores = _normal_scores(supports)
    sides = _atom_sides(node_xy, calibration, lab.shape[1], lab.shape[0])
    line_dirs = _axis_line_directions(node_xy, calibration,
                                      lab.shape[1], lab.shape[0])
    return (_AtomGraph(labels=labels, lab=lab, usable=usable,
                       node_area=node_area, node_lab=node_lab, node_xy=node_xy,
                       node_texture=node_texture,
                       supports=supports, plane_scores=scores, sides=sides,
                       axis_line_dirs=line_dirs,
                       adjacency=adjacency), evidence)


def _seed_and_grow(graph: _AtomGraph, axis: int, side: int,
                   config: RegionGrowthConfig,
                   trace: dict[str, int] | None = None) -> list[list[int]]:
    count = len(graph.node_area)
    scores = graph.plane_scores
    best = scores[:, axis]
    other = np.max(np.delete(scores, axis, axis=1), axis=1)
    tangent = [a for a in range(3) if a != axis]
    seed_support = graph.supports[:, tangent].sum(axis=1)
    tangent_support = graph.supports[:, tangent]
    tangent_total = tangent_support.sum(axis=1)
    normal_support = graph.supports[:, axis]
    tangent_dominance = tangent_total / np.maximum(
        tangent_total + normal_support, 1e-6)
    # A planar continuation can be locally supported by only one of its two
    # Manhattan tangent directions. Requiring a balanced two-axis score for
    # every atom strands truncated and texture-poor portions of a real wall.
    # Keep the seed score balanced, but allow growth from a continuous tangent
    # vote only when evidence for the candidate normal stays weak.
    continuation = tangent_support.max(axis=1) * tangent_dominance
    continuation_ok = ((continuation >= config.min_continuation_support)
                       & (tangent_dominance >= config.min_tangent_dominance))
    seed = ((best >= config.min_seed_score)
            & ((best - other) >= config.min_seed_margin)
            & (seed_support >= config.min_seed_support)
            & (graph.sides[:, axis] == side))
    observer = _attrition_state.get()
    if observer is not None:
        with _scalar_observation_guard(observer):
            identity = (_attrition_frame.get(), axis, side)
            side_match = graph.sides[:, axis] == side
            score_pass = best >= config.min_seed_score
            margin = best - other
            margin_pass = margin >= config.min_seed_margin
            support_pass = seed_support >= config.min_seed_support
            after_score = side_match & score_pass
            after_margin = after_score & margin_pass

            def finite_side_max(values):
                if not np.any(side_match):
                    return None
                maximum = float(np.max(values[side_match]))
                return maximum if math.isfinite(maximum) else None

            observer.expect("seed", identity)
            observer.finish("seed", identity, "available" if np.any(seed) else "absent",
                            axis=axis, side=side, eligible_seed_count=int(np.count_nonzero(seed)),
                            node_count=count,
                            side_match_count=int(np.count_nonzero(side_match)),
                            normal_score_pass_count=int(np.count_nonzero(score_pass)),
                            margin_pass_count=int(np.count_nonzero(margin_pass)),
                            tangent_support_pass_count=int(np.count_nonzero(support_pass)),
                            after_side_count=int(np.count_nonzero(side_match)),
                            after_score_count=int(np.count_nonzero(after_score)),
                            after_margin_count=int(np.count_nonzero(after_margin)),
                            after_support_count=int(np.count_nonzero(after_margin & support_pass)),
                            max_side_normal_score=finite_side_max(best),
                            max_side_margin=finite_side_max(margin),
                            max_side_tangent_support=finite_side_max(seed_support))
            seed_sink = getattr(observer, "seed_predicates", None)
            if seed_sink is not None:
                seed_sink(identity, count=count, best=best, other=other,
                          seed_support=seed_support, sides=graph.sides[:, axis],
                          seed=seed, config=config)
    if not np.any(seed):
        if trace is not None:
            trace["no_seeds"] = trace.get("no_seeds", 0) + 1
        return []

    # Seed appearance is a candidate-plane-specific anchor for the wall color.
    # Quantization and a deterministic palette cap bound the nearest-color
    # comparison on textured, high-resolution frames.
    seed_palette = np.unique(
        np.rint(graph.node_lab[seed] / 4.0).astype(np.int16), axis=0
    ).astype(np.float32) * 4.0
    if len(seed_palette) > 512:
        palette_indices = np.linspace(0, len(seed_palette) - 1, 512, dtype=int)
        seed_palette = seed_palette[palette_indices]
    seed_appearance_distance = np.empty(count, dtype=np.float32)
    for start in range(0, count, 128):
        stop = min(count, start + 128)
        delta = graph.node_lab[start:stop, None, :] - seed_palette[None, :, :]
        seed_appearance_distance[start:stop] = np.sqrt(
            np.min(np.einsum("ijk,ijk->ij", delta, delta), axis=1))

    allowed = ((graph.sides[:, axis] == side)
               & ((best >= config.min_growth_score) | continuation_ok
                  | (graph.supports.sum(axis=1) < config.min_bridge_support)))
    visited = np.zeros(count, dtype=bool)
    regions = []

    def can_cross(source: int, target: int, edge_gradient: float,
                  color_delta: float, texture_delta: float,
                  boundary_tangent_x: float, boundary_tangent_y: float) -> bool:
        source_score = float(best[source])
        target_score = float(best[target])
        target_other = float(other[target])
        target_total = float(graph.supports[target].sum())
        target_continuation = float(continuation[target])
        target_dominance = float(tangent_dominance[target])
        source_other = float(other[source])
        source_continuation = float(continuation[source])
        source_dominance = float(tangent_dominance[source])
        strong_same_plane = (
            source_score >= config.min_seed_score
            and target_score >= config.min_seed_score
            and target_score + config.min_seed_margin >= target_other
        )
        tangent_vector = np.asarray((boundary_tangent_x, boundary_tangent_y),
                                    dtype=np.float32)
        source_dirs = graph.axis_line_dirs[source, tangent]
        target_dirs = graph.axis_line_dirs[target, tangent]
        alignment = max(float(np.max(np.abs(source_dirs @ tangent_vector))),
                        float(np.max(np.abs(target_dirs @ tangent_vector))))
        competing_plane = (target_other >= config.min_seed_score
                           and target_other >= target_score + config.min_seed_margin)
        source_competing = (source_other >= config.min_seed_score
                            and source_other >= source_score + config.min_seed_margin)
        structural_continuation = (
            alignment >= config.min_boundary_axis_alignment
            and not competing_plane
            and not source_competing
            and (
                (target_continuation >= config.min_continuation_support
                 and target_dominance >= config.min_tangent_dominance)
                or (source_continuation >= config.min_continuation_support
                    and source_dominance >= config.min_tangent_dominance
                    and seed_appearance_distance[target] <= config.max_bridge_lab_delta)
            )
        )
        boundary_continuation = (
            alignment >= config.min_boundary_axis_alignment
            and not competing_plane
            and not source_competing
            and color_delta < config.max_lab_delta
        )
        # Manhattan-aligned block seams support continuation along a plane.
        # Lab contrast remains a hard occluder barrier even when the boundary
        # orientation matches one of the candidate plane's tangent axes.
        if (edge_gradient >= config.hard_edge_barrier and not strong_same_plane
                and not boundary_continuation and not structural_continuation):
            if trace is not None:
                trace["hard_edge"] = trace.get("hard_edge", 0) + 1
                if alignment < config.min_boundary_axis_alignment:
                    trace["hard_edge_misaligned"] = trace.get("hard_edge_misaligned", 0) + 1
                elif competing_plane:
                    trace["hard_edge_competing"] = trace.get("hard_edge_competing", 0) + 1
                elif color_delta >= config.max_lab_delta:
                    trace["hard_edge_lab"] = trace.get("hard_edge_lab", 0) + 1
            return False
        if color_delta >= config.max_lab_delta and not strong_same_plane:
            if structural_continuation:
                pass
            else:
                if trace is not None:
                    trace["lab_barrier"] = trace.get("lab_barrier", 0) + 1
                return False
        if (texture_delta >= 28.0 and not strong_same_plane
                and not boundary_continuation and not structural_continuation):
            if trace is not None:
                trace["texture_barrier"] = trace.get("texture_barrier", 0) + 1
            return False
        if target_total < config.min_bridge_support:
            accepted = ((color_delta <= config.max_bridge_lab_delta
                         and texture_delta <= 18.0
                         and edge_gradient <= config.bridge_edge_barrier)
                        or boundary_continuation or structural_continuation)
        else:
            accepted = ((target_score >= config.min_growth_score
                         and target_score + 0.08 >= target_other)
                        or (target_continuation >= config.min_continuation_support
                            and target_dominance >= config.min_tangent_dominance)
                        or (boundary_continuation
                            and target_dominance >= config.min_tangent_dominance)
                        or structural_continuation)
        if trace is not None and structural_continuation:
            trace["structural_continuation"] = trace.get("structural_continuation", 0) + 1
            if color_delta >= config.max_lab_delta:
                trace["structural_lab_override"] = trace.get("structural_lab_override", 0) + 1
        if not accepted and trace is not None:
            trace["evidence_barrier"] = trace.get("evidence_barrier", 0) + 1
        return accepted

    for start in np.flatnonzero(seed):
        start = int(start)
        if visited[start]:
            continue
        queue = deque([start])
        visited[start] = True
        members = []
        has_seed = False
        while queue:
            node = queue.popleft()
            members.append(node)
            has_seed |= bool(seed[node])
            for (neighbor, edge_gradient, color_delta, texture_delta,
                 boundary_tangent_x, boundary_tangent_y) in graph.adjacency[node]:
                if visited[neighbor]:
                    if trace is not None:
                        trace["visited_neighbor"] = trace.get("visited_neighbor", 0) + 1
                    continue
                if not allowed[neighbor]:
                    if trace is not None:
                        trace["neighbor_not_allowed"] = trace.get("neighbor_not_allowed", 0) + 1
                    continue
                if trace is not None:
                    trace["edge_attempts"] = trace.get("edge_attempts", 0) + 1
                if can_cross(node, neighbor, edge_gradient, color_delta, texture_delta,
                             boundary_tangent_x, boundary_tangent_y):
                    if trace is not None:
                        trace["accepted_edges"] = trace.get("accepted_edges", 0) + 1
                    visited[neighbor] = True
                    queue.append(neighbor)
        if has_seed:
            regions.append(members)
    if trace is not None:
        trace["regions"] = len(regions)
    return regions


def _component_polygon(mask: np.ndarray) -> tuple[tuple[float, float], ...]:
    polygon = _mask_polygon(mask)
    if len(polygon) > 96:
        stride = int(math.ceil(len(polygon) / 96.0))
        polygon = polygon[::stride]
    return polygon


def _region_key(proposal: PlaneProposal, width: int, height: int) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(proposal.mask)
    x = float(xs.mean()) if len(xs) else width / 2.0
    y = float(ys.mean()) if len(ys) else height / 2.0
    column = min(2, max(0, int(3 * x / max(width, 1))))
    row = min(2, max(0, int(3 * y / max(height, 1))))
    return proposal.axis, proposal.side, row, column


def _select_diverse(proposals: list[PlaneProposal], width: int, height: int,
                    cap: int) -> tuple[PlaneProposal, ...]:
    """Round-robin spatial/side groups, prioritising measured support.

    When the cap cannot cover every group, axis-number order must not discard
    a stronger supported group in favour of a weaker one. No missing direction
    is synthesised or reserved a slot: only admitted, post-dedup proposals take
    part. Spatial cell, side and axis provide deterministic score tie breaks.
    """
    proposals.sort(key=lambda p: (-p.strength, p.axis, p.side,
                                  _region_key(p, width, height), p.sources))
    groups: dict[tuple[int, int, int, int], list[PlaneProposal]] = {}
    for proposal in proposals:
        groups.setdefault(_region_key(proposal, width, height), []).append(proposal)
    group_order = sorted(groups, key=lambda key: (
        -groups[key][0].strength, key[2], key[3], key[1], key[0]))
    selected: list[PlaneProposal] = []
    index = {key: 0 for key in group_order}
    while len(selected) < cap:
        changed = False
        for key in group_order:
            cursor = index[key]
            if cursor < len(groups[key]):
                selected.append(groups[key][cursor])
                index[key] += 1
                changed = True
                if len(selected) == cap:
                    break
        if not changed:
            break
    return tuple(selected)


def propose_image_planes_v2(
    image: np.ndarray,
    usable_mask: np.ndarray,
    calibration: CalibrationResult | None,
    *,
    max_candidates: int = MAX_REGION_GROWTH_PROPOSALS,
    config: RegionGrowthConfig = DEFAULT_REGION_GROWTH_CONFIG,
    _support_line_weight_policy: str = "structural",
    _atom_snapshot_observer=None,
    _diagnostic_frame_binding=None,
    _diagnostic_source_image=None,
) -> PlaneProposalResult:
    """Grow deterministic image-only plane masks and adapt them to the shared API."""
    weight_policy = _validate_support_line_weight_policy(_support_line_weight_policy)
    started = time.perf_counter()
    if not isinstance(config, RegionGrowthConfig):
        return _empty("invalid_config")
    if (config.atom_spacing < 1 or config.slic_iterations < 1
            or config.max_atoms < 1
            or not math.isfinite(config.connectivity_min_component_fraction)
            or not 0.0 <= config.connectivity_min_component_fraction <= 1.0):
        return _empty("invalid_config")
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
        raise ValueError("image must be a 2D or 3D numpy array")
    rgb = _as_rgb8(image)
    height, width = rgb.shape[:2]
    usable = np.asarray(usable_mask, dtype=bool)
    if usable.shape != (height, width):
        raise ValueError("usable_mask must match the image height and width")
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    cap = min(int(max_candidates), config.max_candidates, MAX_REGION_GROWTH_PROPOSALS)
    usable_fraction = float(usable.mean())
    stats = {
        "usable_fraction": usable_fraction,
        "input_pixels": int(height * width),
        "analysis_pixels": 0,
        "seed_count": 0,
        "components_before_cleanup": 0,
        "connectivity_min_component_area": 0,
        "connectivity_cleanup_merges": 0,
        "isolated_small_components_retained": 0,
        "components_after_cleanup": 0,
        "connectivity_fallback_assignment_pixels": 0,
        "connectivity_mask_preserved": False,
        "atoms": 0,
        "segments": 0,
        "segments_detected": 0,
        "segments_clipped": 0,
        "segments_with_axis_assignment": 0,
        "atoms_with_support": 0,
        "support_mass": 0.0,
        "seed_atoms": 0,
        "seed_atoms_by_axis": (0, 0, 0),
        "graph_edges": 0,
        "proposals_before_cap": 0,
        "proposals_returned": 0,
        "elapsed_ms": 0.0,
    }
    if calibration is None:
        return _empty("missing_calibration", stats)
    if not isinstance(calibration, CalibrationResult):
        return _empty("invalid_calibration", stats)
    numeric = (calibration.focal_pixels, calibration.yaw_degrees,
               calibration.pitch_degrees, calibration.roll_degrees,
               calibration.median_residual)
    principal_valid = (calibration.principal_point is None
                       or (len(calibration.principal_point) == 2
                           and all(np.isfinite(v) for v in calibration.principal_point)))
    if (not all(np.isfinite(v) for v in numeric) or not principal_valid
            or calibration.focal_pixels <= 0
            or calibration.width != width or calibration.height != height):
        return _empty("invalid_or_mismatched_calibration", stats)
    if calibration.inliers < config.min_calibration_inliers:
        return _empty("insufficient_calibration_inliers", stats)
    if usable_fraction < config.min_usable_fraction:
        return _empty("insufficient_usable_coverage", stats)
    if height * width > config.max_input_pixels:
        return _empty("input_pixel_cap", stats)

    prepared = _working_image(rgb, usable, config)
    if prepared is None:
        return _empty("input_pixel_cap", stats)
    _rgb_small, grey, lab, usable_small, scale_x, scale_y = prepared
    work_height, work_width = grey.shape
    stats["analysis_pixels"] = int(work_height * work_width)
    seed_count = (int(math.ceil(work_height / config.atom_spacing))
                  * int(math.ceil(work_width / config.atom_spacing)))
    stats["seed_count"] = seed_count
    if seed_count > config.max_atoms:
        return _empty("atom_cap", stats)
    small_calibration = CalibrationResult(
        focal_pixels=calibration.focal_pixels / scale_x,
        yaw_degrees=calibration.yaw_degrees,
        pitch_degrees=calibration.pitch_degrees,
        width=work_width,
        height=work_height,
        inliers=calibration.inliers,
        segments=calibration.segments,
        median_residual=calibration.median_residual,
        per_axis_inliers=calibration.per_axis_inliers,
        roll_degrees=calibration.roll_degrees,
        principal_point=(None if calibration.principal_point is None else
                         (calibration.principal_point[0] / scale_x,
                          calibration.principal_point[1] / scale_y)),
        notes=calibration.notes,
    )
    atomized = _local_lab_xy_atoms(
        lab, usable_small, spacing=config.atom_spacing,
        iterations=config.slic_iterations, compactness=config.slic_compactness,
        min_component_area_fraction=config.connectivity_min_component_fraction,
        return_stats=True,
    )
    if not isinstance(atomized, tuple) or len(atomized) != 2:
        return _empty("atom_connectivity_invariant_failed", stats)
    labels, atom_stats = atomized
    stats.update(atom_stats)
    atom_count = int(labels.max()) + 1 if labels.size and labels.max() >= 0 else 0
    stats["atoms"] = atom_count
    if not stats["connectivity_mask_preserved"]:
        return _empty("atom_connectivity_invariant_failed", stats)
    if atom_count == 0:
        return _empty("no_usable_atoms", stats)
    if atom_count > config.max_atoms:
        return _empty("atom_connectivity_cap", stats)
    if _atom_snapshot_observer is not None:
        try:
            native_image = image if _diagnostic_source_image is None else _diagnostic_source_image
            if not np.array_equal(_as_rgb8(native_image), rgb):
                raise ValueError("diagnostic RGB adaptation mismatch")
            snapshot = _diagnostic_atom_snapshot(native_image, usable, labels, usable_small,
                (scale_x, scale_y), _diagnostic_frame_binding)
            _atom_snapshot_observer(snapshot)
        except Exception:
            try:
                _atom_snapshot_observer.discard("atom_snapshot_incomplete")
            except Exception:
                pass
    graph, evidence = _build_graph(labels, grey, lab, usable_small,
                                   small_calibration, config,
                                   original_pixel_scale=(scale_x, scale_y),
                                   _support_line_weight_policy=weight_policy)
    if graph is None:
        return _empty("insufficient_atoms", stats)
    stats.update(evidence)
    stats["segments"] = int(evidence.get("segments_clipped", 0))
    stats["graph_edges"] = int(sum(len(edges) for edges in graph.adjacency) // 2)
    seed_best = np.max(graph.plane_scores, axis=1)
    stats["seed_atoms"] = int(np.count_nonzero(seed_best >= config.min_seed_score))
    stats["seed_atoms_by_axis"] = tuple(
        int(np.count_nonzero(graph.plane_scores[:, axis] >= config.min_seed_score))
        for axis in range(3))
    if stats["segments"] == 0:
        stats["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
        return _empty("no_structural_segments", stats)
    if stats["seed_atoms"] == 0:
        stats["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
        return _empty("insufficient_plane_seeds", stats)

    provisional: list[PlaneProposal] = []
    observer = _attrition_state.get()
    generated_ordinal = 0
    usable_count = max(int(usable_small.sum()), 1)
    min_region_area = max(64, int(work_height * work_width
                                  * config.min_region_fraction))
    for axis in range(3):
        for side in (-1, 1):
            for members in _seed_and_grow(graph, axis, side, config):
                identity = (_attrition_frame.get(), generated_ordinal)
                generated_ordinal += 1
                if observer is not None:
                    with _scalar_observation_guard(observer):
                        observer.expect("proposal", identity)
                member_array = np.asarray(members, dtype=np.int32)
                member_map = np.zeros(atom_count, dtype=bool)
                member_map[member_array] = True
                mask_small = (labels >= 0) & member_map[np.maximum(labels, 0)] & usable_small
                area = int(mask_small.sum())
                if area < min_region_area:
                    if observer is not None:
                        with _scalar_observation_guard(observer):
                            observer.finish("proposal", identity, "mask_empty" if area == 0 else "area_rejected",
                                            axis=axis, side=side, area=area)
                    continue
                score = graph.plane_scores[member_array, axis]
                seed_score = float(np.max(score)) if score.size else 0.0
                area_factor = math.sqrt(min(1.0, area / float(max(min_region_area * 12, 1))))
                strength = float(np.clip((0.65 * float(np.mean(score))
                                          + 0.35 * seed_score) * area_factor, 0.0, 1.0))
                if (height, width) != (work_height, work_width):
                    import cv2
                    full_mask = cv2.resize(mask_small.astype(np.uint8), (width, height),
                                           interpolation=cv2.INTER_NEAREST) > 0
                else:
                    full_mask = mask_small.copy()
                full_mask &= usable
                polygon = _component_polygon(full_mask)
                if len(polygon) < 3:
                    if observer is not None:
                        with _scalar_observation_guard(observer):
                            observer.finish("proposal", identity, "polygon_rejected",
                                            axis=axis, side=side, area=area)
                    continue
                full_mask.setflags(write=False)
                provisional.append(PlaneProposal(
                    axis=axis,
                    side=side,
                    polygon=polygon,
                    mask=full_mask,
                    strength=strength,
                    sources=("lab_xy_edge_graph_growth_v2",),
                    coverage=float(area / usable_count),
                ))
                if observer is not None:
                    with _scalar_observation_guard(observer):
                        observer.bind(("proposal", _attrition_frame.get(), id(provisional[-1])), identity)

    provisional.sort(key=lambda p: (-p.strength, p.axis, p.side,
                                    _region_key(p, width, height)))
    deduplicated: list[PlaneProposal] = []
    for proposal in provisional:
        match = next((i for i, old in enumerate(deduplicated)
                      if old.axis == proposal.axis and old.side == proposal.side
                      and _mask_iou(old.mask, proposal.mask) >= config.dedup_iou), None)
        if match is None:
            deduplicated.append(proposal)
        else:
            old = deduplicated[match]
            if proposal.strength > old.strength:
                deduplicated[match] = proposal
                rejected = old
            else:
                rejected = proposal
            if observer is not None:
                with _scalar_observation_guard(observer):
                    observer.finish("proposal", observer.bindings.get(("proposal", _attrition_frame.get(), id(rejected))), "deduplicated",
                                    axis=int(rejected.axis), side=int(rejected.side))
    stats["proposals_before_cap"] = len(deduplicated)
    selected = _select_diverse(deduplicated, width, height, cap)
    if observer is not None:
        with _scalar_observation_guard(observer):
            selected_ordinals = {id(proposal): ordinal for ordinal, proposal in enumerate(selected)}
            for proposal in deduplicated:
                ordinal = selected_ordinals.get(id(proposal))
                generated_id = observer.bindings.get(("proposal", _attrition_frame.get(), id(proposal)))
                observer.finish("proposal", generated_id,
                                "returned" if ordinal is not None else "diversity_cap",
                                axis=int(proposal.axis), side=int(proposal.side), returned_ordinal=ordinal)
                if ordinal is not None and generated_id is not None:
                    observer.bind(("proposal_return", *generated_id), (_attrition_frame.get(), ordinal))
            frame = _attrition_frame.get()
            census = observer._stage("proposal")
            generated = {key for key, value in census["expected"].items() if value[0] == frame}
            completed = {key for key in census["terminal"] if json.loads(key)[0] == frame}
            observer.check("proposal_generation:" + frame,
                           len(generated) == generated_ordinal and generated == completed
                           and all(len(census["expected"][key]) == 2
                                   and type(census["expected"][key][1]) is int
                                   and 0 <= census["expected"][key][1] < generated_ordinal
                                   for key in generated))
            returned = [value for key, value in observer.bindings.items()
                        if key[0] == "proposal_return" and key[1] == frame]
            observer.check("proposal_returned:" + frame,
                           len(returned) == len(selected)
                           and set(returned) == {(frame, ordinal) for ordinal in range(len(selected))})
    stats["proposals_returned"] = len(selected)
    stats["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
    if not selected:
        return _empty("no_grown_plane_regions", stats)
    return PlaneProposalResult(status="ok", reason="image_edge_graph_growth",
                               proposals=selected, stats=stats)
