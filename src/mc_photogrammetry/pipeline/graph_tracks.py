"""Graph-wide multi-view keypoint tracks (DS-25, research-only).

DS-14/18/24 all decide a pose from the correspondences of **one pair**. DS-16B
and DS-24 showed that the surviving pairwise loss is dominated by repeated
texture: a reciprocal or directed top-k candidate graph contains many
descriptor-plausible but geometrically wrong edges, and no pairwise one-to-one
rule can tell a repeated-texture mass from a consistent minority.

DS-25 asks the complementary question: **does a track that is simultaneously
consistent in three or more views resolve the repetition ambiguity?** The answer
is built from three ingredients only, none of which is a reference label:

* the **frozen directed top-k candidate graphs** of a complete session, built
  with the existing descriptor settings (``k = 5``, no distance ceiling, no
  retuning) - see :func:`build_directed_graphs`;
* **closed 3-cycles on exact keypoint identities** across three distinct frames,
  filtered by the frozen descriptor ratio test - see :func:`closed_cycles`;
* a **label-blind score** that uses only normalized descriptor distances, cycle
  support and image-derived epipolar residuals of the M3 yaw/pitch/roll
  estimates - see :func:`track_closure` and :func:`resolve_conflicts`.

The module reads pixels/features, M3 camera orientation and the session's frame
names. It reads no manifest, litematic, scene transform, reference class, oracle
edge or evaluator output; nothing it returns is conditioned on an outcome.

Research-only: no production path imports this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from mc_photogrammetry.pipeline.correspondence import FrameFeatures
from mc_photogrammetry.pipeline.directed_matching import (
    DirectedCandidateGraph,
    directed_topk_graph,
)
from mc_photogrammetry.reconstruction.camera import CameraModel, camera_rotation_matrix

__all__ = [
    "DEFAULT_NEIGHBORS",
    "DEFAULT_MAX_EDGES",
    "DEFAULT_RATIO_TEST",
    "DEFAULT_EPIPOLAR_DEGREES",
    "SessionView",
    "GraphTracksResult",
    "Track",
    "build_directed_graphs",
    "closed_cycles",
    "resolve_conflicts",
    "track_closure",
    "track_score",
    "select_tracks",
    "tracks_to_matches",
    "tracks_to_records",
]

#: Frozen descriptor settings. These are DS-14/DS-18's declared bounds, not new
#: numbers: the candidate graph is the existing directed top-k graph.
DEFAULT_NEIGHBORS = 5
DEFAULT_MAX_EDGES = 8192
#: The frozen ``CorrespondenceSettings.ratio_test`` (0.75); used as the
#: per-frame-pair ambiguity guard on the drawn cycle edges.
DEFAULT_RATIO_TEST = 0.75
#: Frozen DS-11/DS-16B epipolar inlier residual, in degrees.
DEFAULT_EPIPOLAR_DEGREES = 1.0

_TINY = 1e-12


@dataclass(frozen=True, slots=True)
class SessionView:
    """One session frame: identity, features, image-derived camera, image size.

    ``camera`` is a manifest-free :class:`CameraModel` whose ``position`` is
    irrelevant; only yaw/pitch/roll are read (the M3 estimate carries no
    position). ``features`` carries the frozen keypoint pixels/descriptors.
    """

    name: str
    features: FrameFeatures
    camera: CameraModel
    intrinsics: tuple[tuple[float, float], tuple[float, float]]
    """``((focal_x, focal_y), (centre_x, centre_y))`` - pixels only."""

    @property
    def width(self) -> int:
        return int(self.features.width)

    @property
    def height(self) -> int:
        return int(self.features.height)


@dataclass(frozen=True, slots=True)
class Track:
    """One resolved graph-wide track (one keypoint per view)."""

    views: tuple[str, ...]
    points: tuple[int, ...]
    cycles: int
    descriptor_cost: float
    epipolar_residual_degrees: float
    cycle_support_ratio: float
    epipolar_test: float
    score: float
    pixels: tuple[tuple[float, float], ...]

    @property
    def cardinality(self) -> int:
        return len(self.views)

    def point_of(self, view: str) -> int | None:
        for name, point in zip(self.views, self.points):
            if name == view:
                return int(point)
        return None


@dataclass(frozen=True, slots=True)
class GraphTracksResult:
    """A session's frozen graphs and its resolved tracks (deterministic)."""

    session: str
    frames: tuple[str, ...]
    graphs: Mapping[tuple[str, str], DirectedCandidateGraph]
    qualified: Mapping[tuple[str, str], np.ndarray]
    tracks: tuple[Track, ...]
    candidates: int
    discarded_conflicts: int
    discarded_unclosed: int


# --------------------------------------------------------------------------- #
# Candidate graphs.
# --------------------------------------------------------------------------- #
def build_directed_graphs(
    views: Sequence[SessionView],
    *,
    neighbors: int = DEFAULT_NEIGHBORS,
    max_edges: int = DEFAULT_MAX_EDGES,
) -> dict[tuple[str, str], DirectedCandidateGraph]:
    """Directed left->right top-k graphs for every ordered frame pair.

    The graphs are exactly the DS-18 ``directed_topk_graph`` with its declared
    settings; no threshold is introduced, no descriptor is re-detected and no
    reciprocity is required. Both directions of every unordered pair are built so
    the cycle rule is closed under reversing a cycle's orientation.
    """
    graphs: dict[tuple[str, str], DirectedCandidateGraph] = {}
    for left in views:
        for right in views:
            if left.name == right.name:
                continue
            graphs[(left.name, right.name)] = directed_topk_graph(
                left.features.descriptors,
                right.features.descriptors,
                neighbors=neighbors,
                max_edges=max_edges,
                distance_ceiling=None,
            )
    return graphs


def _local_edges(
    graph: DirectedCandidateGraph, left_points: int, right_points: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Graph edges restricted to local index ranges, plus per-left ratio guards.

    Returns ``(edges, distances, second_best, scale)`` where ``edges`` are local
    ``(left, right)`` indices, ``distances`` their descriptor L2 distances,
    ``second_best`` the distance of each left keypoint's second candidate (inf
    when it has only one), and ``scale`` the distance of each left keypoint's
    farthest drawn candidate (its local descriptor-neighbourhood scale, inf when
    the keypoint has no candidate). The ratio test is *not* applied here; callers
    decide whether an edge must be ratio-qualified.
    """
    edges = np.asarray(graph.edges, dtype=np.int64).reshape(-1, 2)
    distances = np.asarray(graph.distances, dtype=np.float64).reshape(-1)
    if len(edges) == 0:
        return (
            np.empty((0, 2), dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.full(left_points, np.inf, dtype=np.float64),
            np.full(left_points, np.inf, dtype=np.float64),
        )
    keep = (edges[:, 0] >= 0) & (edges[:, 0] < left_points)
    keep &= (edges[:, 1] >= 0) & (edges[:, 1] < right_points)
    edges, distances = edges[keep], distances[keep]
    second = np.full(left_points, np.inf, dtype=np.float64)
    scale = np.full(left_points, np.inf, dtype=np.float64)
    order = np.lexsort((edges[:, 1], distances, edges[:, 0]))
    ordered_left = edges[order, 0]
    if len(ordered_left):
        unique, first, counts = np.unique(ordered_left, return_index=True, return_counts=True)
        second[unique[counts >= 2]] = distances[order][first[counts >= 2] + 1]
        scale[unique] = distances[order][first + counts - 1]
    return edges, distances, second, scale


def _ratio_qualified(
    edges: np.ndarray, distances: np.ndarray, second: np.ndarray, ratio: float
) -> np.ndarray:
    """Boolean per-edge ratio-test mask: ``d <= ratio * second_best``."""
    if len(edges) == 0:
        return np.zeros(0, dtype=bool)
    with np.errstate(invalid="ignore"):
        return distances <= ratio * second[edges[:, 0]]


def _edge_lookup(
    edges: np.ndarray,
) -> dict[tuple[int, int], int]:
    return {(int(left), int(right)): row for row, (left, right) in enumerate(edges)}


# --------------------------------------------------------------------------- #
# Closed 3-cycles.
# --------------------------------------------------------------------------- #
def _cycle(
    frame_a: str,
    frame_b: str,
    frame_c: str,
    point_a: int,
    point_b: int,
    point_c: int,
    lookups: Mapping[tuple[str, str], Mapping[tuple[int, int], int]],
) -> tuple[tuple[str, int], ...] | None:
    """A closed 3-cycle in the qualified graphs, in canonical orientation.

    Requires all six directed edges ``a->b``, ``b->a``, ``b->c``, ``c->b``,
    ``c->a``, ``a->c`` to be ratio-qualified candidate edges, i.e. every vertex's
    keypoint is each other vertex's descriptor neighbour *and* unambiguous under
    the frozen ratio test. Returns the three ``(frame, point)`` vertices or
    ``None``.
    """
    checks = (
        ((frame_a, frame_b), (point_a, point_b)),
        ((frame_b, frame_a), (point_b, point_a)),
        ((frame_b, frame_c), (point_b, point_c)),
        ((frame_c, frame_b), (point_c, point_b)),
        ((frame_c, frame_a), (point_c, point_a)),
        ((frame_a, frame_c), (point_a, point_c)),
    )
    for key, (left, right) in checks:
        table = lookups.get(key)
        if table is None or (left, right) not in table:
            return None
    return ((frame_a, point_a), (frame_b, point_b), (frame_c, point_c))


def closed_cycles(
    views: Sequence[SessionView],
    graphs: Mapping[tuple[str, str], DirectedCandidateGraph],
    *,
    ratio_test: float = DEFAULT_RATIO_TEST,
    max_cycles: int | None = None,
) -> list[tuple[tuple[str, int], ...]]:
    """Every closed 3-cycle on exact keypoint identities, deterministically.

    Cycles are enumerated over ascending frame triples and ascending keypoint
    triples, so the returned order is a pure function of the *content* of the
    session and does not depend on frame ordering or frame labels. A cycle is
    only admitted when all six directed edges are ratio-qualified; a frame
    triple with fewer than three frames is skipped and frames are distinct.
    """
    qualified: dict[tuple[str, str], np.ndarray] = {}
    lookups: dict[tuple[str, str], Mapping[tuple[int, int], int]] = {}
    for key, graph in graphs.items():
        left_view = next(view for view in views if view.name == key[0])
        right_view = next(view for view in views if view.name == key[1])
        edges, distances, second, _scale = _local_edges(
            graph, len(left_view.features.points), len(right_view.features.points)
        )
        mask = _ratio_qualified(edges, distances, second, ratio_test)
        keep = edges[mask]
        qualified[key] = keep
        lookups[key] = _edge_lookup(keep)

    cycles: list[tuple[tuple[str, int], ...]] = []
    frames = [view.name for view in views]
    for first in range(len(frames)):
        for second in range(first + 1, len(frames)):
            for third in range(second + 1, len(frames)):
                frame_a, frame_b, frame_c = frames[first], frames[second], frames[third]
                ab = lookups[(frame_a, frame_b)]
                if not ab:
                    continue
                tables = {
                    (frame_a, frame_b): ab,
                    (frame_b, frame_c): lookups[(frame_b, frame_c)],
                    (frame_c, frame_a): lookups[(frame_c, frame_a)],
                    (frame_b, frame_a): lookups[(frame_b, frame_a)],
                    (frame_c, frame_b): lookups[(frame_c, frame_b)],
                    (frame_a, frame_c): lookups[(frame_a, frame_c)],
                }
                if not all(tables.values()):
                    continue
                # Iterate ``a->b`` edges (the smallest pair) and close the cycle
                # through the two remaining qualified edges.
                for (point_a, point_b) in sorted(ab.keys()):
                    for point_c in _cycle_targets(
                        tables[(frame_b, frame_c)], point_b,
                        tables[(frame_c, frame_a)], point_a,
                    ):
                        found = _cycle(
                            frame_a, frame_b, frame_c,
                            point_a, point_b, point_c, tables,
                        )
                        if found is not None:
                            cycles.append(found)
                            if max_cycles is not None and len(cycles) >= max_cycles:
                                return cycles
    return cycles


def _cycle_targets(
    bc: Mapping[tuple[int, int], int],
    point_b: int,
    ca: Mapping[tuple[int, int], int],
    point_a: int,
) -> list[int]:
    """Right endpoints of ``b->c`` edges from ``point_b`` that also close via ``a``."""
    targets = [right for (left, right) in bc if left == point_b]
    return [right for right in targets if (right, point_a) in ca]


# --------------------------------------------------------------------------- #
# Label-blind closure and score.
# --------------------------------------------------------------------------- #
def _pixel_ray(pixel: Sequence[float], intrinsics) -> np.ndarray:
    focal_x, focal_y = intrinsics[0]
    centre_x, centre_y = intrinsics[1]
    ray = np.asarray(
        (
            (float(pixel[0]) - centre_x) / focal_x,
            -(float(pixel[1]) - centre_y) / focal_y,
            1.0,
        ),
        dtype=np.float64,
    )
    return ray / np.linalg.norm(ray)


def _baseline_direction(rows: np.ndarray) -> np.ndarray | None:
    """Bounded Gram/SVD null vector (DS-18/DS-19 mechanism, no RANSAC)."""
    if rows.size == 0:
        return None
    eigenvalues, vectors = np.linalg.eigh(rows.T @ rows)
    direction = vectors[:, 0]
    norm = float(np.linalg.norm(direction))
    if norm <= _TINY:
        return None
    return direction / norm


def track_closure(
    view_of: Mapping[str, SessionView],
    points: Mapping[str, int],
    pairs: Sequence[tuple[str, str]] | None = None,
) -> tuple[float, float, float]:
    """Epipolar residual of one track under the image-derived M3 geometry.

    For every unordered view pair of the track, the per-pair baseline direction
    is the bounded 3x3 Gram/SVD null vector of that pair's ``d_left x d_right``
    rows (the DS-18/DS-19 mechanism, no sampling); each endpoint's residual is
    the angle between its epipolar plane normal and that direction. Returns
    ``(max_residual_degrees, mean_residual_degrees, coverage)`` where coverage is
    the fraction of pairs with at least one informative row.
    """
    names = sorted(points)
    if pairs is None:
        pairs = [
            (names[first], names[second])
            for first in range(len(names))
            for second in range(first + 1, len(names))
        ]
    residuals: list[float] = []
    informative = 0
    for left_name, right_name in pairs:
        left = view_of[left_name]
        right = view_of[right_name]
        left_ray = _pixel_ray(left.features.points[points[left_name]], left.intrinsics)
        right_ray = _pixel_ray(right.features.points[points[right_name]], right.intrinsics)
        rotation_left = camera_rotation_matrix(left.camera)
        rotation_right = camera_rotation_matrix(right.camera)
        rows: list[np.ndarray] = []
        for yaw_class in range(4):
            relative = (rotation_right @ _ry90(yaw_class)) @ rotation_left.T
            cross = np.cross(left_ray @ relative.T, right_ray)
            if float(np.linalg.norm(cross)) > 1e-9:
                rows.append(cross / float(np.linalg.norm(cross)))
        if not rows:
            residuals.append(float("inf"))
            continue
        informative += 1
        direction = _baseline_direction(np.asarray(rows, dtype=np.float64))
        if direction is None:
            residuals.append(float("inf"))
            continue
        # Worst over the four yaw hypotheses: a false track gains no benefit.
        worst = float("inf")
        for row in rows:
            projection = float(np.clip(abs(float(np.dot(row, direction))), 0.0, 1.0))
            worst = min(worst, float(np.degrees(np.arcsin(projection))))
        residuals.append(worst)
    if not pairs:
        return (float("inf"), float("inf"), 0.0)
    values = np.asarray(residuals, dtype=np.float64)
    finite = values[np.isfinite(values)]
    maximum = float(values.max()) if np.all(np.isfinite(values)) else float("inf")
    mean = float(finite.mean()) if len(finite) else float("inf")
    coverage = informative / float(len(pairs))
    return (maximum, mean, coverage)


def _ry90(yaw_class: int) -> np.ndarray:
    angle = np.radians(90.0 * (yaw_class % 4))
    cosine, sine = float(np.cos(angle)), float(np.sin(angle))
    return np.asarray(
        ((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine)), dtype=np.float64
    )


def track_score(
    *,
    cycles: int,
    possible_cycles: int,
    descriptor_cost: float,
    epipolar_residual_degrees: float,
    epipolar_degrees: float = DEFAULT_EPIPOLAR_DEGREES,
) -> tuple[float, float, float]:
    """The preregistered label-blind score of one track.

    Three terms, each in ``[0, 1]``, with a fixed lexicographic priority encoded
    by the same weights and by the deterministic tie-break in
    :func:`select_tracks`:

    ``support``
        ``cycles / possible_cycles`` - how many independent closed 3-cycles the
        track closes.
    ``geometry``
        ``max(0, 1 - worst_epipolar_residual / epipolar_degrees)`` - the frozen
        DS-11 epipolar inlier law applied to the image-derived M3 geometry.
    ``descriptor``
        ``max(0, 1 - descriptor_cost)`` where ``descriptor_cost`` is the mean
        descriptor L2 distance of the drawn cycle edges normalized by that
        frame pair's own nearest-candidate distance.

    Returns ``(score, geometry_term, support_term)``.
    """
    support = cycles / float(possible_cycles) if possible_cycles > 0 else 0.0
    geometry = max(0.0, 1.0 - float(epipolar_residual_degrees) / float(epipolar_degrees))
    descriptor = max(0.0, 1.0 - float(descriptor_cost))
    return (
        float(min(support, 1.0) + geometry + descriptor),
        float(geometry),
        float(min(support, 1.0)),
    )


# --------------------------------------------------------------------------- #
# Candidate tracks and the global conflict resolution.
# --------------------------------------------------------------------------- #
@dataclass
class _Candidate:
    views: tuple[str, ...]
    points: tuple[int, ...]
    cycles: int
    descriptor_cost: float
    epipolar: float
    closure_mean: float
    coverage: float
    possible_cycles: int
    score: float
    geometry: float
    pixels: tuple[tuple[float, float], ...]


def _qualified_tables(
    views: Sequence[SessionView],
    graphs: Mapping[tuple[str, str], DirectedCandidateGraph],
    ratio_test: float,
) -> tuple[
    dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]],
    dict[tuple[str, str], Mapping[tuple[int, int], int]],
]:
    """Ratio-qualified ``(edges, distances, scale)`` and their ``(l, r) -> row``."""
    kept: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    tables: dict[tuple[str, str], Mapping[tuple[int, int], int]] = {}
    counts = {view.name: len(view.features.points) for view in views}
    for key, graph in graphs.items():
        edges, distances, second, scale = _local_edges(
            graph, counts[key[0]], counts[key[1]]
        )
        mask = _ratio_qualified(edges, distances, second, ratio_test)
        keep = edges[mask]
        kept[key] = (keep, distances[mask], scale)
        tables[key] = _edge_lookup(keep)
    return kept, tables


def _cycles_for_track(
    points: Mapping[str, int],
    tables: Mapping[tuple[str, str], Mapping[tuple[int, int], int]],
    *,
    cap: int | None = None,
) -> int:
    """Count closed 3-cycles among a track's own points in the frozen graphs."""
    names = sorted(points)
    count = 0
    for first in range(len(names)):
        for second in range(first + 1, len(names)):
            for third in range(second + 1, len(names)):
                triple = (names[first], names[second], names[third])
                if _cycle(
                    triple[0], triple[1], triple[2],
                    points[triple[0]], points[triple[1]], points[triple[2]],
                    tables,
                ) is not None:
                    count += 1
                    if cap is not None and count >= cap:
                        return count
    return count


def select_tracks(
    views: Sequence[SessionView],
    graphs: Mapping[tuple[str, str], DirectedCandidateGraph],
    *,
    ratio_test: float = DEFAULT_RATIO_TEST,
    epipolar_degrees: float = DEFAULT_EPIPOLAR_DEGREES,
    max_cycles: int | None = None,
) -> GraphTracksResult:
    """Build every closed-cycle track and resolve competing ones globally.

    A candidate track is the union of the keypoints of one or more closed
    3-cycles that share keypoints (a chain of overlapping cycles); competing
    tracks that share a keypoint in any view are resolved greedily, best score
    first, with the deterministic keys ``(score desc, epipolar residual asc,
    descriptor cost asc, view names, point indices)``. The score is a fixed
    function of the frozen graph content, so no reference class, manifest or
    evaluator output can influence the choice.
    """
    view_of = {view.name: view for view in views}
    cycles = closed_cycles(
        views, graphs, ratio_test=ratio_test, max_cycles=max_cycles
    )
    kept, tables = _qualified_tables(views, graphs, ratio_test)

    # Merge cycles that share a keypoint into connected track components.
    parent: list[int] = []

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        root_a, root_b = find(first), find(second)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    owner: dict[tuple[str, int], int] = {}
    for index, cycle in enumerate(cycles):
        parent.append(index)
        for vertex in cycle:
            if vertex in owner:
                union(index, owner[vertex])
            else:
                owner[vertex] = index

    groups: dict[int, set[tuple[str, int]]] = {}
    for index, cycle in enumerate(cycles):
        groups.setdefault(find(index), set()).update(cycle)

    candidates: list[_Candidate] = []
    discarded_unclosed = 0
    for vertices in groups.values():
        if len(vertices) < 3:
            discarded_unclosed += 1
            continue
        points = {name: point for name, point in vertices}
        names = sorted(points)
        possible = len(names) * (len(names) - 1) * (len(names) - 2) // 6
        epipolar, mean, coverage = track_closure(view_of, points)
        if not np.isfinite(epipolar) or epipolar > epipolar_degrees:
            discarded_unclosed += 1
            continue
        cost = _descriptor_cost(kept, tables, points)
        if cost is None:
            discarded_unclosed += 1
            continue
        closed = _cycles_for_track(points, tables, cap=possible)
        if closed < 1:
            discarded_unclosed += 1
            continue
        score, geometry, _support = track_score(
            cycles=closed,
            possible_cycles=possible,
            descriptor_cost=cost,
            epipolar_residual_degrees=epipolar,
            epipolar_degrees=epipolar_degrees,
        )
        pixels = tuple(
            (float(view_of[name].features.points[points[name]][0]),
             float(view_of[name].features.points[points[name]][1]))
            for name in names
        )
        candidates.append(
            _Candidate(
                views=tuple(names),
                points=tuple(int(points[name]) for name in names),
                cycles=closed,
                descriptor_cost=float(cost),
                epipolar=float(epipolar),
                closure_mean=float(mean),
                coverage=float(coverage),
                possible_cycles=int(possible),
                score=float(score),
                geometry=float(geometry),
                pixels=pixels,
            )
        )

    ordered = sorted(
        candidates,
        key=lambda item: (
            -item.score,
            item.epipolar,
            item.descriptor_cost,
            item.views,
            item.points,
        ),
    )
    claimed: dict[str, set[int]] = {name: set() for name in view_of}
    survivors: list[_Candidate] = []
    discarded_conflicts = 0
    for candidate in ordered:
        if any(
            point in claimed[name]
            for name, point in zip(candidate.views, candidate.points)
        ):
            discarded_conflicts += 1
            continue
        for name, point in zip(candidate.views, candidate.points):
            claimed[name].add(point)
        survivors.append(candidate)

    tracks = tuple(
        Track(
            views=candidate.views,
            points=candidate.points,
            cycles=candidate.cycles,
            descriptor_cost=candidate.descriptor_cost,
            epipolar_residual_degrees=candidate.epipolar,
            cycle_support_ratio=(
                candidate.cycles / candidate.possible_cycles
                if candidate.possible_cycles
                else 0.0
            ),
            epipolar_test=max(
                0.0, 1.0 - candidate.epipolar / float(epipolar_degrees)
            ),
            score=candidate.score,
            pixels=candidate.pixels,
        )
        for candidate in survivors
    )
    return GraphTracksResult(
        session="",
        frames=tuple(sorted(view_of)),
        graphs=graphs,
        qualified={key: value[0] for key, value in kept.items()},
        tracks=tracks,
        candidates=len(cycles),
        discarded_conflicts=discarded_conflicts,
        discarded_unclosed=discarded_unclosed,
    )


def _descriptor_cost(
    kept: Mapping[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]],
    tables: Mapping[tuple[str, str], Mapping[tuple[int, int], int]],
    points: Mapping[str, int],
) -> float | None:
    """Mean normalized descriptor distance of the track's drawn cycle edges.

    Every drawn directed edge is normalized by the originating left keypoint's
    own descriptor-neighbourhood scale (the distance of its farthest frozen
    top-k candidate), so the measure is a per-edge rank in ``[0, 1]`` that is
    comparable across frame pairs. Only edges that are ratio-qualified count; a
    track without a fully drawn cycle contributes ``None``.
    """
    names = sorted(points)
    drawn: list[float] = []
    for left_name in names:
        for right_name in names:
            if left_name == right_name:
                continue
            edges, distances, scale = kept[(left_name, right_name)]
            if len(edges) == 0:
                return None
            row = tables[(left_name, right_name)].get(
                (int(points[left_name]), int(points[right_name]))
            )
            if row is None:
                return None
            reference = float(scale[int(points[left_name])])
            if not np.isfinite(reference) or reference <= _TINY:
                drawn.append(1.0)
            else:
                drawn.append(float(distances[row]) / reference)
    if not drawn:
        return None
    return float(np.mean(np.clip(drawn, 0.0, 1.0)))


def resolve_conflicts(
    views: Sequence[SessionView],
    graphs: Mapping[tuple[str, str], DirectedCandidateGraph],
    **kwargs,
) -> GraphTracksResult:
    """Alias for :func:`select_tracks` (the preregistered resolver)."""
    return select_tracks(views, graphs, **kwargs)


# --------------------------------------------------------------------------- #
# Hand-over helpers.
# --------------------------------------------------------------------------- #
def tracks_to_matches(
    tracks: Sequence[Track], left: str, right: str
) -> np.ndarray:
    """Pixel correspondences ``(left_x, left_y, right_x, right_y)`` from tracks.

    Every track that carries a keypoint in ``left`` and in ``right`` contributes
    exactly one correspondence, so the derived set is one-to-one by construction
    and a repeated-texture mass cannot inflate a pair independently of the
    graph-wide resolution. The rows are sorted by ``(left pixel, right pixel)``
    which is identity-free and order-stable.
    """
    rows: list[tuple[float, float, float, float]] = []
    for track in tracks:
        left_point = track.point_of(left)
        right_point = track.point_of(right)
        if left_point is None or right_point is None:
            continue
        rows.append(
            (
                float(track.pixels[track.views.index(left)][0]),
                float(track.pixels[track.views.index(left)][1]),
                float(track.pixels[track.views.index(right)][0]),
                float(track.pixels[track.views.index(right)][1]),
            )
        )
    if not rows:
        return np.empty((0, 4), dtype=np.float64)
    table = np.asarray(rows, dtype=np.float64)
    order = np.lexsort((table[:, 2], table[:, 3], table[:, 0], table[:, 1]))
    return table[order]


def tracks_to_records(tracks: Sequence[Track]) -> list[dict]:
    """Identity-complete, order-stable serialization of the resolved tracks."""
    return [
        {
            "views": list(track.views),
            "points": [int(value) for value in track.points],
            "pixels": [[round(float(x), 6), round(float(y), 6)] for x, y in track.pixels],
            "cardinality": int(track.cardinality),
            "cycles": int(track.cycles),
            "possible_cycles": int(
                track.cardinality * (track.cardinality - 1) * (track.cardinality - 2) // 6
            ),
            "cycle_support_ratio": round(float(track.cycle_support_ratio), 9),
            "descriptor_cost": round(float(track.descriptor_cost), 9),
            "epipolar_residual_degrees": round(
                float(track.epipolar_residual_degrees), 9
            ),
            "epipolar_test": round(float(track.epipolar_test), 9),
            "score": round(float(track.score), 9),
        }
        for track in tracks
    ]
