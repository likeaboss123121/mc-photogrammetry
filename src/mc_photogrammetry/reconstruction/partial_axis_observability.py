"""Source-bound axis connectivity, without coordinates or geometry authority.

An anchor-connected component has a resolved *relative* gauge. Every other
component retains its independent unknown world offset, including singletons.
The observable core is the intersection of the three anchor components.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import plane_graph

TRANSCRIPT_POLICY = "source-bound-partial-axis-transcript-v2-ac3"


@dataclass(frozen=True)
class AxisObservability:
    anchor: str
    core: tuple[str, ...]
    # (world axis, ((component frames, relative gauge frame, anchor connected), ...))
    partitions: tuple

    def payload(self):
        return (self.anchor, self.core, self.partitions)

    def resolved_axes(self, frame):
        return tuple(axis for axis, components in self.partitions
                     if any(known and frame in members for members, _gauge, known in components))


def select_anchor(work, indexed, frames):
    """Charge inventory work before retaining the deterministic anchor census."""
    work.charge("relation_visits", len(frames) + len(indexed))
    axes = {frame: set() for frame in frames}
    for key, reading in indexed.items():
        axes[key[0]].add(reading.axis)
    return next((frame for frame in sorted(frames) if axes[frame] == {0, 1, 2}), None)


def classify(work, factors, indexed, frames, quadrants, anchor):
    """All active factors remain mandatory, including floating peripherals."""
    if not factors:
        return "no_active_factors", None
    if anchor is None:
        return "no_fully_observed_anchor", None
    work.charge("relation_visits", 3 * len(frames))
    adjacency = {axis: {frame: set() for frame in frames} for axis in range(3)}
    supported = set()
    for u, v, _relation in factors:
        work.charge("relation_visits", 1)
        supported.update((u, v))
        axis, _ = plane_graph.axis_transform(indexed[u].axis, quadrants[u[0]])
        other, _ = plane_graph.axis_transform(indexed[v].axis, quadrants[v[0]])
        if axis != other:
            return "active_axis_mismatch", None
        adjacency[axis][u[0]].add(v[0])
        adjacency[axis][v[0]].add(u[0])
    if supported != set(indexed):
        return "unsupported_reading_endpoint", None
    partitions, core = [], set(frames)
    for axis in range(3):
        unseen, components = set(frames), []
        while unseen:
            start = min(unseen)
            visited, queue = {start}, [start]
            while queue:
                work.charge("relation_visits", 1)
                node = queue.pop()
                for neighbour in sorted(adjacency[axis][node]):
                    work.charge("relation_visits", 1)
                    if neighbour not in visited:
                        visited.add(neighbour)
                        queue.append(neighbour)
            unseen.difference_update(visited)
            known = anchor in visited
            components.append((tuple(sorted(visited)), anchor if known else start, known))
            if known:
                core.intersection_update(visited)
        partitions.append((axis, tuple(components)))
    result = AxisObservability(anchor, tuple(sorted(core)), tuple(partitions))
    return (None if len(core) >= 2 else "no_two_frame_observable_core"), result


def from_payload(value):
    """Strict canonical scalar shape; source replay must still verify topology."""
    if type(value) not in (list, tuple) or len(value) != 3:
        raise ValueError("partial axis observability shape differs")
    anchor, core, partitions = value
    def frame_valid(frame):
        return type(frame) is str and 0 < len(frame) <= 4096 and len(frame.encode("utf-8")) <= 4096

    if (not frame_valid(anchor) or type(core) not in (tuple, list) or len(core) > 32
            or any(not frame_valid(frame) for frame in core)
            or list(core) != sorted(set(core)) or len(core) < 2 or anchor not in core
            or type(partitions) not in (tuple, list) or len(partitions) != 3):
        raise ValueError("partial axis core/anchor differs")
    result, universe, intersection = [], None, None
    for expected_axis, row in enumerate(partitions):
        if (type(row) not in (tuple, list) or len(row) != 2
                or type(row[0]) is not int or row[0] != expected_axis
                or type(row[1]) not in (tuple, list) or not 0 < len(row[1]) <= 32):
            raise ValueError("partial axis partition differs")
        components, seen, anchored = [], set(), None
        for component in row[1]:
            if type(component) not in (tuple, list) or len(component) != 3:
                raise ValueError("partial axis component shape differs")
            members, gauge, known = component
            if (type(members) not in (tuple, list) or not 0 < len(members) <= 32
                    or any(not frame_valid(frame) for frame in members)
                    or list(members) != sorted(set(members)) or seen.intersection(members)
                    or type(gauge) is not str or type(known) is not bool
                    or known != (anchor in members)
                    or gauge != (anchor if known else members[0])):
                raise ValueError("partial axis component gauge differs")
            seen.update(members)
            if len(seen) > 32:
                raise ValueError("partial axis component census exceeds frame cap")
            components.append((tuple(members), gauge, known))
            if known:
                anchored = set(members)
        if anchored is None or components != sorted(components):
            raise ValueError("partial axis component order/anchor differs")
        if universe is not None and seen != universe:
            raise ValueError("partial axis frame census differs")
        universe = seen
        intersection = anchored if intersection is None else intersection.intersection(anchored)
        result.append((expected_axis, tuple(components)))
    if set(core) != intersection:
        raise ValueError("partial axis core is not the component intersection")
    return AxisObservability(anchor, tuple(core), tuple(result))
