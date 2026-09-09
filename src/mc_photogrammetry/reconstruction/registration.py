"""Camera translations from planes two or more views agree they share.

`supported.py` places a plane in one view's own local frame. It has no way to
relate that frame to another view's: a single image fixes no position, only
shape and (with a stated length) distance. Two views only become comparable once
something says they are looking at the *same* physical surface.

That something cannot be appearance. `supported.py` measured phase agreement -
the obvious appearance-based test for "is this cell really a block face" - at
chance across four decades of scale, on a plane whose distance was independently
right to 0.1%. Automatic cross-view correspondence built on the same kind of
signal would fail for the same reason: Minecraft's repeated textures are exactly
what appearance-matching cannot see through.

So plane identity across views is a stated fact, the same way a length is: cheap
to give, needs no feature matching, and survives the repetition that defeats
appearance. A user (or, in production, an assumption the pipeline is told to
make and can be told to retract) says "the floor in image A and the floor in
image B are the same floor." Given that, and each view's own distance to that
plane, the two eyes' positions along the plane's axis are related exactly:

    eye_a[axis] + side_a * distance_a == eye_b[axis] + side_b * distance_b

because both sides name the same real number - the plane's world coordinate.
Chaining these across a set of links solves every linked view's position along
that axis, up to one free constant per connected component (the axis has no
absolute origin without one more fact, e.g. a stated world coordinate for one
plane). That is the honest amount of information the inputs provide, and no
more: two views that share no asserted plane on an axis stay unrelated on it,
and the result says so rather than guessing at zero.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PlaneObservation:
    """One view's reading of a plane's distance, in that view's own frame."""

    view: str
    axis: int
    side: float
    distance: float


@dataclass(frozen=True)
class SharedPlaneLink:
    """A stated fact, not a measurement: two views' planes are the same surface.

    This is the cross-view analogue of the stated length in `scale_from_known_length`
    - the one constraint Minecraft's repetition cannot defeat, because nothing here
    infers it from what the pictures look like.
    """

    view_a: str
    view_b: str
    axis: int


@dataclass(frozen=True)
class AxisRegistration:
    """One connected component of views tied together by shared-plane links."""

    axis: int
    anchor: str
    positions: dict[str, float]
    """Each linked view's eye coordinate on this axis, relative to the anchor."""
    redundant_links: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    """Links beyond a spanning tree: present for transparency, not validation.

    With exactly one observation per (view, axis) - the case this module is built
    for - a link's equation is a plain difference of two fixed per-view numbers,
    and any cycle those differences form sums to zero identically, for *any*
    values at all. That is provable rather than merely usually true: it holds
    even when a link asserts two views share a plane they manifestly do not.
    So a redundant link can never disagree with the tree under this data model,
    and finding one that does would mean a bug in this function, not a bad
    assertion in the input - which is exactly why the tests check that this list
    stays empty of disagreement rather than exercising a "wrong link" case.

    A wrong shared-plane assertion is therefore invisible to this module by
    construction. Catching one needs evidence from outside the link graph: a
    third kind of measurement, or - during development - the reference. See
    also `scale_from_known_length`, which carries the same shaped limitation for
    a single stated length ("this cannot check it").
    """


def register_axis(
    observations: list[PlaneObservation],
    links: list[SharedPlaneLink],
    axis: int,
    anchor: str,
) -> AxisRegistration | None:
    """Solve eye positions on one axis from observations tied by shared-plane links.

    ``anchor`` is fixed at 0.0; every other linked view's position is relative to
    it, because nothing in the inputs names an absolute origin. Returns ``None``
    if the anchor made no observation on this axis, or is not reachable from any
    link. Raises if a view supplies more than one observation on this axis: a
    dict keyed by view would silently keep the last and drop the rest, and this
    module has no basis for choosing between two disagreeing readings of what a
    view saw.
    """
    by_view: dict[str, PlaneObservation] = {}
    for obs in observations:
        if obs.axis != axis:
            continue
        if obs.view in by_view:
            raise ValueError(
                f"{obs.view!r} supplied more than one observation on axis "
                f"{axis}; register_axis cannot choose between them")
        by_view[obs.view] = obs
    if anchor not in by_view:
        return None

    edges: dict[str, list[tuple[str, float]]] = {}
    for link in links:
        if link.axis != axis:
            continue
        if link.view_a not in by_view or link.view_b not in by_view:
            continue
        a, b = by_view[link.view_a], by_view[link.view_b]
        # eye_a[axis] + side_a*distance_a == eye_b[axis] + side_b*distance_b, so
        # eye_b[axis] - eye_a[axis] == side_a*distance_a - side_b*distance_b.
        delta = a.side * a.distance - b.side * b.distance
        edges.setdefault(a.view, []).append((b.view, delta))
        edges.setdefault(b.view, []).append((a.view, -delta))

    positions = {anchor: 0.0}
    tree_edges: set[tuple[str, str]] = set()
    frontier = [anchor]
    while frontier:
        current = frontier.pop()
        for neighbour, delta in edges.get(current, []):
            if neighbour in positions:
                continue
            positions[neighbour] = positions[current] + delta
            tree_edges.add(frozenset((current, neighbour)))
            frontier.append(neighbour)

    redundant = []
    seen: set[frozenset] = set()
    for link in links:
        if link.axis != axis:
            continue
        if link.view_a not in positions or link.view_b not in positions:
            continue
        pair = frozenset((link.view_a, link.view_b))
        if pair in tree_edges or pair in seen:
            continue
        seen.add(pair)
        redundant.append((link.view_a, link.view_b))

    return AxisRegistration(axis=axis, anchor=anchor, positions=positions,
                            redundant_links=tuple(redundant))
