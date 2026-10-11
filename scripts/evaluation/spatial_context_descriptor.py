"""Synthetic-only spatial-context descriptor A/B (32D central vs 288D row-major context).

Evaluation prototype implementing the sealed Astra contract
``RESULT-ASTRA-SPATIAL-CONTEXT-DESCRIPTOR-CONTRACT-01-2026-10-09.json``.

Boundaries
----------
* Pure NumPy/Python; the public API consumes in-memory arrays only.
* No CLI, image ingress, authorization, output publication, or production import.
* Never opens, hashes, decodes, or otherwise touches screenshots/masks on disk.
* No OpenCV/SIFT, GPU, inference, or native-image matching.
* Deterministic under input-enumeration permutation; per-site identity is stable
  and binds frame/source digests, exact mask digests, centre, 48px footprint and
  the descriptor contract version.
* A and B are deterministic reconstructions of one shared ``(N, 9, 32)`` float32
  tile table; controls consume authored pixels, never prebuilt descriptor vectors.

Exact RGB-to-luma, tiling, gradient, histogram, normalization, matching and
common-four witness semantics follow the Astra receipt without reinterpretation.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import itertools
import json
import math
from typing import Iterable, Sequence

import numpy as np

CONTRACT_VERSION = "spatial-context-ab-v1"
SITE_SCHEMA = "spatial-context-site-v1"
DESCRIPTOR_SCHEMA = "spatial-context-arm-descriptor-v1"
PAIR_SCHEMA = "spatial-context-reciprocal-pair-v1"
BINDING_SCHEMA = "spatial-context-source-binding-v1"

SPACING = 16
OFFSET = 8
WINDOW = 48
HALF_WINDOW = 24
WINDOW_PIXELS = WINDOW * WINDOW
TILES_PER_AXIS = 3
TILE = 16
CELLS_PER_AXIS = 2
ORIENTATION_BINS = 8
TILE_DIM = CELLS_PER_AXIS * CELLS_PER_AXIS * ORIENTATION_BINS
B_DIM = TILES_PER_AXIS * TILES_PER_AXIS * TILE_DIM
CENTRAL_TILE_INDEX = 4
GRADIENT_SAMPLES_PER_AXIS = TILE - 2

RATIO = 0.75
RATIO_SQUARED = RATIO * RATIO

LEFT_SITE_LIMIT = 256
RIGHT_SITE_LIMIT = 4096
TOTAL_SITE_LIMIT = 4352
COMPARISON_LIMIT = 1048576
WITNESS_NODE_LIMIT = 4096
WITNESS_DETERMINANT_LIMIT = 32768

SUPPORT_BINS_PER_AXIS = 4
DISTANCE_BLOCK_ROWS = 8

FLOAT32_LE = np.dtype("<f4")
FLOAT64 = np.dtype("<f8")
BOOL = np.dtype(bool)
UINT8 = np.dtype(np.uint8)


class Refusal(RuntimeError):
    """Terminal input/integrity/resource refusal; never a negative finding."""


# ---------------------------------------------------------------------------
# Canonical primitives
# ---------------------------------------------------------------------------


def canonical(value) -> bytes:
    """Canonical compact ASCII JSON used for every digest."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def validate_rgb(pixels) -> tuple[int, int]:
    """Strict uint8 HxWx3 finite native raster; reject rather than convert."""
    if not isinstance(pixels, np.ndarray):
        raise Refusal("pixels_not_ndarray")
    if pixels.dtype != UINT8:
        raise Refusal("pixels_not_uint8")
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise Refusal("pixels_shape")
    return int(pixels.shape[0]), int(pixels.shape[1])


def validate_mask(mask, shape: tuple[int, int], name: str) -> None:
    if not isinstance(mask, np.ndarray):
        raise Refusal(name + "_not_ndarray")
    if mask.dtype != BOOL:
        raise Refusal(name + "_not_bool")
    if mask.shape != shape:
        raise Refusal(name + "_shape")


def validate_descriptor_array(array, dim: int) -> None:
    if not isinstance(array, np.ndarray):
        raise Refusal("descriptor_not_ndarray")
    if array.dtype != FLOAT32_LE:
        raise Refusal("descriptor_not_little_endian_float32")
    if array.ndim != 2 or array.shape[1] != dim:
        raise Refusal("descriptor_shape")
    if not np.isfinite(array).all():
        raise Refusal("nonfinite_descriptor")


def dimension_for_arm(arm: str) -> int:
    if arm == "A":
        return TILE_DIM
    if arm == "B":
        return B_DIM
    raise Refusal("invalid_arm")


# ---------------------------------------------------------------------------
# Exact luma, tiling, gradient, histogram
# ---------------------------------------------------------------------------


def luma_from_rgb(pixels) -> np.ndarray:
    """diagnostic-defined integer luma: Y=(77R+150G+29B+128)//256 then /255."""
    validate_rgb(pixels)
    rgb = pixels.astype(np.uint32)
    promoted = 77 * rgb[:, :, 0] + 150 * rgb[:, :, 1] + 29 * rgb[:, :, 2] + 128
    integer = promoted // 256
    return integer.astype(np.float64) / 255.0


def tile_gradients(tile_luma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Local gx/gy for u,v=1..14 inside one 16x16 tile; signed float arithmetic."""
    tile = np.asarray(tile_luma, dtype=np.float64)
    if tile.shape != (TILE, TILE):
        raise Refusal("tile_shape")
    if not np.isfinite(tile).all():
        raise Refusal("nonfinite_tile")
    gx = tile[1:-1, 2:] - tile[1:-1, :-2]
    gy = tile[2:, 1:-1] - tile[:-2, 1:-1]
    return gx, gy


def tile_histogram(tile_luma: np.ndarray) -> np.ndarray:
    """32D float64 histogram (cell row, cell col, orientation bin) in v-then-u order."""
    gx, gy = tile_gradients(tile_luma)
    magnitude = np.hypot(gx, gy)
    histogram = np.zeros(TILE_DIM, dtype=np.float64)
    for v_index in range(GRADIENT_SAMPLES_PER_AXIS):
        v = v_index + 1
        for u_index in range(GRADIENT_SAMPLES_PER_AXIS):
            u = u_index + 1
            weight = float(magnitude[v_index, u_index])
            if weight == 0.0:
                continue  # zero-magnitude samples contribute nothing
            theta = math.atan2(float(gy[v_index, u_index]), float(gx[v_index, u_index])) % math.pi
            t = ORIENTATION_BINS * theta / math.pi
            k = int(math.floor(t))
            alpha = t - k
            cell_row = 0 if v < 8 else 1
            cell_col = 0 if u < 8 else 1
            base = (cell_row * CELLS_PER_AXIS + cell_col) * ORIENTATION_BINS
            histogram[base + (k % ORIENTATION_BINS)] += weight * (1.0 - alpha)
            histogram[base + ((k + 1) % ORIENTATION_BINS)] += weight * alpha
    if not np.isfinite(histogram).all():
        raise Refusal("nonfinite_histogram")
    return histogram


def normalize_tile_vector(histogram: np.ndarray) -> tuple[np.ndarray, bool]:
    """L2-normalize float64, else keep exact zeros and flag the zero tile."""
    if histogram.shape != (TILE_DIM,):
        raise Refusal("histogram_shape")
    norm_squared = float(np.dot(histogram, histogram))
    if norm_squared > 0.0:
        normalized = histogram / math.sqrt(norm_squared)
    else:
        normalized = histogram.copy()
    if not np.isfinite(normalized).all():
        raise Refusal("nonfinite_tile_vector")
    stored = np.asarray(normalized, dtype=FLOAT32_LE)
    return stored, bool(norm_squared == 0.0)


def tile_table(luma: np.ndarray, x: int, y: int) -> tuple[np.ndarray, tuple[bool, ...]]:
    """3x3 nonoverlapping 16x16 tiles plus zero flags for one 48x48 window."""
    window = luma[y - HALF_WINDOW : y + HALF_WINDOW, x - HALF_WINDOW : x + HALF_WINDOW]
    if window.shape != (WINDOW, WINDOW):
        raise Refusal("window_bounds")
    tiles = np.empty((TILES_PER_AXIS * TILES_PER_AXIS, TILE_DIM), dtype=FLOAT32_LE)
    zero_flags: list[bool] = []
    for index in range(TILES_PER_AXIS * TILES_PER_AXIS):
        tile_y, tile_x = divmod(index, TILES_PER_AXIS)
        tile = window[tile_y * TILE : (tile_y + 1) * TILE, tile_x * TILE : (tile_x + 1) * TILE]
        stored, is_zero = normalize_tile_vector(tile_histogram(tile))
        tiles[index] = stored
        zero_flags.append(is_zero)
    if not np.isfinite(tiles).all():
        raise Refusal("nonfinite_tile_table")
    return tiles, tuple(zero_flags)


def build_context_b(tiles: np.ndarray) -> np.ndarray:
    """288D: concatenate the nine stored 32D tiles row-major, L2-normalize, cast once."""
    if tiles.shape != (TILES_PER_AXIS * TILES_PER_AXIS, TILE_DIM):
        raise Refusal("tile_table_shape")
    flat = tiles.reshape(-1).astype(np.float64)
    norm_squared = float(np.dot(flat, flat))
    if norm_squared > 0.0:
        flat = flat / math.sqrt(norm_squared)
    if not np.isfinite(flat).all():
        raise Refusal("nonfinite_context")
    return np.asarray(flat, dtype=FLOAT32_LE)


# ---------------------------------------------------------------------------
# Site planning and identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Site:
    site_id: str
    x: int
    y: int

    @property
    def xy(self) -> tuple[int, int]:
        return (self.x, self.y)


@dataclass(frozen=True)
class DomainPlan:
    contract_version: str
    shape: tuple[int, int]
    frame_sha256: str
    source_mask_sha256: str
    usable_mask_sha256: str
    grid_centers: int
    eligible_count: int
    sites: tuple[Site, ...]


def _site_id(frame_sha256, source_mask_sha256, usable_mask_sha256, x, y) -> str:
    return sha(
        canonical(
            {
                "schema": SITE_SCHEMA,
                "contract_version": CONTRACT_VERSION,
                "frame_sha256": frame_sha256,
                "source_mask_sha256": source_mask_sha256,
                "usable_mask_sha256": usable_mask_sha256,
                "center": [int(x), int(y)],
                "footprint_px": WINDOW,
            }
        )
    )


def _window_in_bounds(x: int, y: int, shape: tuple[int, int]) -> bool:
    height, width = shape
    return (
        x - HALF_WINDOW >= 0
        and y - HALF_WINDOW >= 0
        and x + HALF_WINDOW <= width
        and y + HALF_WINDOW <= height
    )


def plan_sites(pixels, source_mask, usable_mask) -> DomainPlan:
    """Prescribed grid centres whose whole 48x48 footprint is true in both masks."""
    height, width = validate_rgb(pixels)
    shape = (height, width)
    validate_mask(source_mask, shape, "source_mask")
    validate_mask(usable_mask, shape, "usable_mask")
    frame_sha256 = sha(pixels.tobytes(order="C"))
    source_mask_sha256 = sha(source_mask.tobytes(order="C"))
    usable_mask_sha256 = sha(usable_mask.tobytes(order="C"))

    grid_centers = 0
    eligible: list[tuple[int, int]] = []
    for y in range(OFFSET, height, SPACING):
        for x in range(OFFSET, width, SPACING):
            grid_centers += 1
            if not _window_in_bounds(x, y, shape):
                continue
            source_window = source_mask[y - HALF_WINDOW : y + HALF_WINDOW, x - HALF_WINDOW : x + HALF_WINDOW]
            usable_window = usable_mask[y - HALF_WINDOW : y + HALF_WINDOW, x - HALF_WINDOW : x + HALF_WINDOW]
            if source_window.all() and usable_window.all():
                eligible.append((x, y))

    ordered = sorted(eligible, key=lambda point: (point[1], point[0]))
    sites = tuple(
        Site(_site_id(frame_sha256, source_mask_sha256, usable_mask_sha256, x, y), x, y)
        for x, y in ordered
    )
    return DomainPlan(
        contract_version=CONTRACT_VERSION,
        shape=shape,
        frame_sha256=frame_sha256,
        source_mask_sha256=source_mask_sha256,
        usable_mask_sha256=usable_mask_sha256,
        grid_centers=grid_centers,
        eligible_count=len(sites),
        sites=sites,
    )


def check_site_caps(left_count: int, right_count: int) -> None:
    if type(left_count) is not int or type(right_count) is not int:
        raise Refusal("invalid_site_count")
    if left_count < 0 or right_count < 0:
        raise Refusal("invalid_site_count")
    if left_count > LEFT_SITE_LIMIT:
        raise Refusal("left_site_limit")
    if right_count > RIGHT_SITE_LIMIT:
        raise Refusal("right_site_limit")
    if left_count + right_count > TOTAL_SITE_LIMIT:
        raise Refusal("total_site_limit")


# ---------------------------------------------------------------------------
# Arm descriptors (A = central 32D, B = 288D context)
# ---------------------------------------------------------------------------


@dataclass
class ArmDescriptors:
    arm: str
    sites: tuple[Site, ...]
    descriptors: np.ndarray
    central_zero: np.ndarray
    neighbor_zero_counts: np.ndarray
    tile_tables: np.ndarray
    frame_sha256: str
    shape: tuple[int, int]

    @property
    def shared_indices(self) -> np.ndarray:
        return np.nonzero(~self.central_zero)[0]

    def descriptor_identities(self) -> tuple[str, ...]:
        return tuple(
            sha(
                canonical(
                    {
                        "schema": DESCRIPTOR_SCHEMA,
                        "contract_version": CONTRACT_VERSION,
                        "arm": self.arm,
                        "site_id": site.site_id,
                        "values": [float(value) for value in self.descriptors[row]],
                    }
                )
            )
            for row, site in enumerate(self.sites)
        )


def describe_arm(pixels, sites: Iterable[Site], arm: str) -> ArmDescriptors:
    """Build the shared tile table and the requested arm descriptor for each site."""
    dimension_for_arm(arm)
    shape = validate_rgb(pixels)
    luma = luma_from_rgb(pixels)
    frame_sha256 = sha(pixels.tobytes(order="C"))
    ordered = tuple(sorted(sites, key=lambda site: (site.y, site.x, site.site_id)))
    identities = [site.site_id for site in ordered]
    if len(set(identities)) != len(identities):
        raise Refusal("duplicate_site_identity")

    dim = dimension_for_arm(arm)
    count = len(ordered)
    descriptors = np.empty((count, dim), dtype=FLOAT32_LE)
    central_zero = np.zeros(count, dtype=bool)
    neighbor_zero_counts = np.zeros(count, dtype=np.int64)
    tile_tables = np.empty((count, TILES_PER_AXIS * TILES_PER_AXIS, TILE_DIM), dtype=FLOAT32_LE)

    for row, site in enumerate(ordered):
        tiles, zero_flags = tile_table(luma, site.x, site.y)
        tile_tables[row] = tiles
        central_zero[row] = zero_flags[CENTRAL_TILE_INDEX]
        neighbor_zero_counts[row] = sum(
            1 for index, is_zero in enumerate(zero_flags) if index != CENTRAL_TILE_INDEX and is_zero
        )
        descriptors[row] = tiles[CENTRAL_TILE_INDEX] if arm == "A" else build_context_b(tiles)

    validate_descriptor_array(descriptors, dim)
    return ArmDescriptors(
        arm=arm,
        sites=ordered,
        descriptors=descriptors,
        central_zero=central_zero,
        neighbor_zero_counts=neighbor_zero_counts,
        tile_tables=tile_tables,
        frame_sha256=frame_sha256,
        shape=shape,
    )


# ---------------------------------------------------------------------------
# Work budgeting
# ---------------------------------------------------------------------------


@dataclass
class Budget:
    limit: int
    requested: int = 0
    charged: int = 0
    processed: int = 0

    def reserve(self, units: int) -> None:
        if type(units) is not int or units < 0:
            raise Refusal("invalid_work_units")
        self.requested += units
        if self.charged + units > self.limit:
            raise Refusal("work_budget_overflow")
        self.charged += units

    def close(self) -> dict:
        return {
            "limit": self.limit,
            "requested": self.requested,
            "charged": self.charged,
            "processed": self.processed,
            "uncharged": self.requested - self.charged,
            "unprocessed": self.charged - self.processed,
        }


# ---------------------------------------------------------------------------
# Distances and directed ratio matching
# ---------------------------------------------------------------------------


def pairwise_squared(left: np.ndarray, right: np.ndarray, budget: Budget) -> np.ndarray:
    """Squared Euclidean on float64-promoted float32 values; charge before work."""
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise Refusal("distance_shape")
    rows, columns = int(left.shape[0]), int(right.shape[0])
    budget.reserve(rows * columns)
    result = np.empty((rows, columns), dtype=np.float64)
    for start in range(0, rows, DISTANCE_BLOCK_ROWS):
        block = left[start : start + DISTANCE_BLOCK_ROWS]
        delta = block[:, None, :] - right[None, :, :]
        result[start : start + block.shape[0]] = np.sum(delta * delta, axis=2)
        budget.processed += int(block.shape[0]) * columns
    if not np.isfinite(result).all():
        raise Refusal("nonfinite_distance")
    return result


def _directed_empty(origin_ids: Sequence[str]) -> dict:
    return {
        "accepted": {},
        "counts": {
            "origins": len(origin_ids),
            "absent_second_neighbor": 0,
            "nearest_distance_tie": 0,
            "ratio_rejected": 0,
            "ratio_accepted": 0,
        },
        "evidence": (),
    }


def directed_matching(matrix: np.ndarray, origin_ids: Sequence[str], target_ids: Sequence[str]) -> dict:
    """Squared-L2 strict ratio 0.75, no tie winner and no unavailable second neighbour."""
    if matrix.shape != (len(origin_ids), len(target_ids)):
        raise Refusal("directed_matrix_shape")
    counts = {
        "origins": len(origin_ids),
        "absent_second_neighbor": 0,
        "nearest_distance_tie": 0,
        "ratio_rejected": 0,
        "ratio_accepted": 0,
    }
    accepted: dict[str, str] = {}
    evidence: list[dict] = []
    target_count = len(target_ids)
    for row, origin_id in enumerate(origin_ids):
        if target_count < 2:
            counts["absent_second_neighbor"] += 1
            evidence.append({"origin": origin_id, "status": "absent_second_neighbor"})
            continue
        candidates = sorted((float(matrix[row, column]), column) for column in range(target_count))
        (nearest_squared, nearest_column), (second_squared, second_column) = candidates[0], candidates[1]
        if nearest_squared == second_squared:
            status = "nearest_distance_tie"
            counts["nearest_distance_tie"] += 1
        elif nearest_squared < RATIO_SQUARED * second_squared:
            status = "ratio_accepted"
            counts["ratio_accepted"] += 1
            accepted[origin_id] = target_ids[nearest_column]
        else:
            status = "ratio_rejected"
            counts["ratio_rejected"] += 1
        evidence.append(
            {
                "origin": origin_id,
                "nearest": target_ids[nearest_column],
                "second": target_ids[second_column],
                "nearest_squared": nearest_squared,
                "second_squared": second_squared,
                "status": status,
            }
        )
    return {"accepted": accepted, "counts": counts, "evidence": tuple(evidence)}


# ---------------------------------------------------------------------------
# Common-four witness
# ---------------------------------------------------------------------------


def general_position(points: Sequence[tuple[int, int]]) -> bool:
    """True when the coordinate set contains four points with no three collinear."""
    unique = sorted({(int(x), int(y)) for x, y in points})
    if len(unique) < 4:
        return False
    fractions = [(Fraction(x), Fraction(y)) for x, y in unique]

    def area(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    a, b = fractions[:2]
    c = next((point for point in fractions[2:] if area(a, b, point)), None)
    if c is None:
        return False
    triangle = (a, b, c)
    edge_points: dict = {}
    for point in fractions:
        if point in triangle:
            continue
        edges = [index for index, (u, v) in enumerate(((a, b), (b, c), (c, a))) if not area(u, v, point)]
        if not edges:
            return True
        edge_points.setdefault(edges[0], point)
        if len(edge_points) >= 2:
            witnesses = triangle + tuple(edge_points.values())
            return any(
                all(area(*triple) for triple in itertools.combinations(quad, 3))
                for quad in itertools.combinations(witnesses, 4)
            )
    return False


def good_four(points: Sequence[tuple[int, int]]) -> bool:
    """Exactly four distinct coordinates, no three collinear, exact Fraction determinants."""
    if len(points) != 4 or len({(int(x), int(y)) for x, y in points}) != 4:
        return False
    normalized = []
    for x, y in points:
        x_value, y_value = float(x), float(y)
        if not math.isfinite(x_value) or not math.isfinite(y_value):
            raise Refusal("nonfinite_witness_coordinate")
        normalized.append((Fraction.from_float(x_value), Fraction.from_float(y_value)))
    determinants = [
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        for a, b, c in itertools.combinations(normalized, 3)
    ]
    return all(value != 0 for value in determinants)


def make_pair_record(left_id, right_id, left_xy, right_xy, distance_squared) -> dict:
    body = {
        "schema": PAIR_SCHEMA,
        "left_id": left_id,
        "right_id": right_id,
        "left_xy": [int(left_xy[0]), int(left_xy[1])],
        "right_xy": [int(right_xy[0]), int(right_xy[1])],
        "distance_squared": float(distance_squared),
    }
    record = dict(body)
    record["pair_id"] = sha(canonical(body))
    return record


def common_four(pairs, node_budget: Budget, determinant_budget: Budget | None = None,
                source_binding_sha256: str | None = None) -> dict:
    """Streaming canonical tuples with honest attempted/charged/completed work.

    Each attempted tuple requests one node. Its eight determinants are reserved
    atomically before evaluation: an insufficient reservation requests eight,
    charges zero, and leaves the charged tuple unprocessed. A refused reservation
    is retained in closure counters and ends the search as budget_inconclusive.
    Work summaries expose the supplied budgets' cumulative counters; work_nodes
    and determinants retain this call's charged-work deltas.
    """
    if len(pairs) > 1024 or len({pair["pair_id"] for pair in pairs}) != len(pairs):
        raise Refusal("common_witness_match_limit_or_identity")
    if determinant_budget is None:
        determinant_budget = Budget(node_budget.limit * 8)
    before_nodes = node_budget.charged
    before_determinants = determinant_budget.charged
    ordered = sorted(pairs, key=lambda pair: pair["pair_id"])

    def result(status: str, quad=()) -> dict:
        return {
            "status": status,
            "work_nodes": node_budget.charged - before_nodes,
            "determinants": determinant_budget.charged - before_determinants,
            "node_work": node_budget.close(),
            "determinant_work": determinant_budget.close(),
            "pair_ids": [pair["pair_id"] for pair in quad],
            "coordinates": [
                {
                    "pair_id": pair["pair_id"],
                    "left_id": pair["left_id"],
                    "right_id": pair["right_id"],
                    "left_xy": pair["left_xy"],
                    "right_xy": pair["right_xy"],
                }
                for pair in quad
            ],
            "source_binding_sha256": source_binding_sha256,
        }

    for quad in itertools.combinations(ordered, 4):
        try:
            node_budget.reserve(1)
        except Refusal as error:
            if str(error) != "work_budget_overflow":
                raise
            return result("budget_inconclusive")
        valid = False
        if (
            len({pair["left_id"] for pair in quad}) == 4
            and len({pair["right_id"] for pair in quad}) == 4
            and len({tuple(pair["left_xy"]) for pair in quad}) == 4
            and len({tuple(pair["right_xy"]) for pair in quad}) == 4
        ):
            try:
                determinant_budget.reserve(8)
            except Refusal as error:
                if str(error) != "work_budget_overflow":
                    raise
                return result("budget_inconclusive")
            left_valid = good_four([pair["left_xy"] for pair in quad])
            right_valid = good_four([pair["right_xy"] for pair in quad])
            determinant_budget.processed += 8
            valid = left_valid and right_valid
        node_budget.processed += 1
        if valid:
            return result("found", quad)
    return result("exhaustive_no_witness")


# ---------------------------------------------------------------------------
# Support observables
# ---------------------------------------------------------------------------


def _support_block(site_ids: Sequence[str], coordinates: Sequence[tuple[int, int]],
                   shape: tuple[int, int]) -> dict:
    height, width = shape
    bins: list[list[str]] = [[] for _ in range(SUPPORT_BINS_PER_AXIS * SUPPORT_BINS_PER_AXIS)]
    for site_id, (x, y) in zip(site_ids, coordinates):
        index = int(y * SUPPORT_BINS_PER_AXIS / height) * SUPPORT_BINS_PER_AXIS + int(
            x * SUPPORT_BINS_PER_AXIS / width
        )
        bins[index].append(site_id)
    for group in bins:
        group.sort()
    flattened = [site_id for group in bins for site_id in group]
    first_four = flattened[:4]
    by_id = dict(zip(site_ids, coordinates))
    first_four_ready = good_four([by_id[site_id] for site_id in first_four]) if len(first_four) == 4 else False
    return {
        "distinct_sites": len(site_ids),
        "distinct_coordinates": len({tuple(coordinate) for coordinate in coordinates}),
        "bins": [list(group) for group in bins],
        "occupied_bins": sum(1 for group in bins if group),
        "first_four_ids": first_four,
        "first_four_ready": first_four_ready,
        "all_domain_general_position": general_position(coordinates),
    }


def support_report(arm: ArmDescriptors, reference: ArmDescriptors) -> dict:
    """All-domain and canonical first-four support reported separately per side."""
    result = {}
    for name, descriptors in (("query", arm), ("reference", reference)):
        indices = descriptors.shared_indices
        ids = [descriptors.sites[int(index)].site_id for index in indices]
        coordinates = [descriptors.sites[int(index)].xy for index in indices]
        all_domain = _support_block(ids, coordinates, descriptors.shape)
        first_ids = all_domain["first_four_ids"]
        by_id = dict(zip(ids, coordinates))
        result[name] = {
            "all_domain": all_domain,
            "first_four": _support_block(first_ids, [by_id[site_id] for site_id in first_ids], descriptors.shape),
        }
    return result


# ---------------------------------------------------------------------------
# Matching and one-arm analysis
# ---------------------------------------------------------------------------


@dataclass
class ArmMatch:
    arm: str
    status: str
    query_domain_ids: tuple[str, ...]
    reference_domain_ids: tuple[str, ...]
    forward_counts: dict
    reverse_counts: dict
    forward_evidence: tuple[dict, ...]
    reverse_evidence: tuple[dict, ...]
    pairs: tuple[dict, ...]
    witness: dict
    comparison_work: dict
    distance_matrix: np.ndarray
    source_binding_sha256: str | None


@dataclass
class ArmAnalysis:
    arm: str
    query_plan: DomainPlan
    reference_plan: DomainPlan
    query_descriptors: ArmDescriptors
    reference_descriptors: ArmDescriptors
    match: ArmMatch
    support: dict


def _validate_arm_layout(arm: ArmDescriptors, dim: int) -> None:
    """Check all capped row alignment without copies or numerical scans."""
    count = len(arm.sites)
    for name, dtype, shape in (
        ("descriptors", FLOAT32_LE, (count, dim)),
        ("central_zero", BOOL, (count,)),
        ("neighbor_zero_counts", np.dtype(np.int64), (count,)),
        ("tile_tables", FLOAT32_LE, (count, 9, TILE_DIM)),
    ):
        array = getattr(arm, name)
        if not isinstance(array, np.ndarray):
            raise Refusal(name + "_not_ndarray")
        if array.dtype != dtype:
            raise Refusal(name + "_dtype")
        if array.shape != shape:
            raise Refusal(name + "_shape")


def match_arms(query: ArmDescriptors, reference: ArmDescriptors, *,
               comparison_limit: int = COMPARISON_LIMIT,
               witness_node_limit: int = WITNESS_NODE_LIMIT,
               witness_determinant_limit: int = WITNESS_DETERMINANT_LIMIT,
               source_binding_sha256: str | None = None) -> ArmMatch:
    check_site_caps(len(query.sites), len(reference.sites))
    if query.arm != reference.arm:
        raise Refusal("arm_mismatch")
    for name, limit, ceiling in (
        ("comparison_limit", comparison_limit, COMPARISON_LIMIT),
        ("witness_node_limit", witness_node_limit, WITNESS_NODE_LIMIT),
        ("witness_determinant_limit", witness_determinant_limit, WITNESS_DETERMINANT_LIMIT),
    ):
        if type(limit) is not int or limit < 0:
            raise Refusal("invalid_" + name)
        if limit > ceiling:
            raise Refusal(name + "_above_frozen_cap")
    dim = dimension_for_arm(query.arm)
    # Both layouts must pass before even the first finite-value scan.
    _validate_arm_layout(query, dim)
    _validate_arm_layout(reference, dim)
    validate_descriptor_array(query.descriptors, dim)
    validate_descriptor_array(reference.descriptors, dim)

    query_indices = query.shared_indices
    reference_indices = reference.shared_indices
    query_ids = [query.sites[int(index)].site_id for index in query_indices]
    reference_ids = [reference.sites[int(index)].site_id for index in reference_indices]
    query_positions = {site_id: position for position, site_id in enumerate(query_ids)}
    reference_positions = {site_id: position for position, site_id in enumerate(reference_ids)}
    coordinates = {}
    for site in query.sites:
        coordinates[site.site_id] = site.xy
    for site in reference.sites:
        coordinates[site.site_id] = site.xy

    comparisons = Budget(comparison_limit)
    if len(query_indices) == 0 or len(reference_indices) == 0:
        matrix = np.zeros((len(query_indices), len(reference_indices)), dtype=np.float64)
        forward = _directed_empty(query_ids)
        reverse = _directed_empty(reference_ids)
    else:
        left = query.descriptors[query_indices].astype(np.float64)
        right = reference.descriptors[reference_indices].astype(np.float64)
        matrix = pairwise_squared(left, right, comparisons)
        forward = directed_matching(matrix, query_ids, reference_ids)
        reverse = directed_matching(matrix.T, reference_ids, query_ids)

    pairs = []
    for left_id in sorted(forward["accepted"]):
        right_id = forward["accepted"][left_id]
        if reverse["accepted"].get(right_id) == left_id:
            distance = float(matrix[query_positions[left_id], reference_positions[right_id]])
            pairs.append(
                make_pair_record(left_id, right_id, coordinates[left_id], coordinates[right_id], distance)
            )
    pairs.sort(key=lambda pair: pair["pair_id"])

    node_budget = Budget(witness_node_limit)
    determinant_budget = Budget(witness_determinant_limit)
    witness = common_four(pairs, node_budget, determinant_budget, source_binding_sha256)

    if len(query_indices) == 0 or len(reference_indices) == 0:
        status = "insufficient_support"
    elif witness["status"] == "found":
        status = "candidate_discrimination_witness"
    elif witness["status"] == "budget_inconclusive":
        status = "witness_budget_inconclusive"
    elif pairs:
        status = "reciprocal_pairs_without_common_witness"
    else:
        status = "no_admitted_pairs"

    return ArmMatch(
        arm=query.arm,
        status=status,
        query_domain_ids=tuple(query_ids),
        reference_domain_ids=tuple(reference_ids),
        forward_counts=forward["counts"],
        reverse_counts=reverse["counts"],
        forward_evidence=forward["evidence"],
        reverse_evidence=reverse["evidence"],
        pairs=tuple(pairs),
        witness=witness,
        comparison_work=comparisons.close(),
        distance_matrix=matrix,
        source_binding_sha256=source_binding_sha256,
    )


def source_binding(query_plan: DomainPlan, reference_plan: DomainPlan) -> str:
    return sha(
        canonical(
            {
                "schema": BINDING_SCHEMA,
                "contract_version": CONTRACT_VERSION,
                "query": {
                    "frame_sha256": query_plan.frame_sha256,
                    "source_mask_sha256": query_plan.source_mask_sha256,
                    "usable_mask_sha256": query_plan.usable_mask_sha256,
                },
                "reference": {
                    "frame_sha256": reference_plan.frame_sha256,
                    "source_mask_sha256": reference_plan.source_mask_sha256,
                    "usable_mask_sha256": reference_plan.usable_mask_sha256,
                },
            }
        )
    )


def analyze_arm(arm: str, query_pixels, query_source_mask, query_usable_mask,
                reference_pixels, reference_source_mask, reference_usable_mask, *,
                comparison_limit: int = COMPARISON_LIMIT,
                witness_node_limit: int = WITNESS_NODE_LIMIT,
                witness_determinant_limit: int = WITNESS_DETERMINANT_LIMIT) -> ArmAnalysis:
    """Plan both frames, enforce site caps, derive one arm, then match it."""
    query_plan = plan_sites(query_pixels, query_source_mask, query_usable_mask)
    reference_plan = plan_sites(reference_pixels, reference_source_mask, reference_usable_mask)
    check_site_caps(len(query_plan.sites), len(reference_plan.sites))
    query_descriptors = describe_arm(query_pixels, query_plan.sites, arm)
    reference_descriptors = describe_arm(reference_pixels, reference_plan.sites, arm)
    match = match_arms(
        query_descriptors,
        reference_descriptors,
        comparison_limit=comparison_limit,
        witness_node_limit=witness_node_limit,
        witness_determinant_limit=witness_determinant_limit,
        source_binding_sha256=source_binding(query_plan, reference_plan),
    )
    return ArmAnalysis(
        arm=arm,
        query_plan=query_plan,
        reference_plan=reference_plan,
        query_descriptors=query_descriptors,
        reference_descriptors=reference_descriptors,
        match=match,
        support=support_report(query_descriptors, reference_descriptors),
    )


__all__ = [
    "ArmAnalysis",
    "ArmDescriptors",
    "ArmMatch",
    "Budget",
    "CONTRACT_VERSION",
    "DomainPlan",
    "Refusal",
    "Site",
    "analyze_arm",
    "build_context_b",
    "canonical",
    "check_site_caps",
    "common_four",
    "describe_arm",
    "dimension_for_arm",
    "directed_matching",
    "general_position",
    "good_four",
    "luma_from_rgb",
    "make_pair_record",
    "match_arms",
    "normalize_tile_vector",
    "pairwise_squared",
    "plan_sites",
    "sha",
    "source_binding",
    "support_report",
    "tile_gradients",
    "tile_histogram",
    "tile_table",
    "validate_descriptor_array",
    "validate_mask",
    "validate_rgb",
]
