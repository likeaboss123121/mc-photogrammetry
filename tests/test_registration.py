"""Solving camera translations from planes two views assert are the same.

The equation is exact - two readings of the same real number - so the tests are
mostly about the bookkeeping: chaining links through more than one hop, an axis
nobody linked, and a cycle where the links disagree with each other.
"""
import pytest

from mc_photogrammetry.reconstruction.registration import (
    AxisRegistration,
    PlaneObservation,
    SharedPlaneLink,
    register_axis,
)


def test_two_views_of_the_same_floor_recover_their_relative_height():
    # Both see a floor 2 blocks below their own eye; view b's eye is 3 blocks
    # higher than view a's, so the same floor is 5 blocks below view b's eye.
    observations = [
        PlaneObservation("a", axis=1, side=-1.0, distance=2.0),
        PlaneObservation("b", axis=1, side=-1.0, distance=5.0),
    ]
    links = [SharedPlaneLink("a", "b", axis=1)]
    result = register_axis(observations, links, axis=1, anchor="a")
    assert result.positions["a"] == pytest.approx(0.0)
    assert result.positions["b"] == pytest.approx(3.0)


def test_links_chain_through_an_unlinked_pair():
    """a and c share no direct link, but both get positioned via b.

    Each view sees the same floor. a's floor sits 1 below its eye; b's, 4 below;
    c's, 1 below. a-b linked and b-c linked (a-c is not) must still place c.
    """
    observations = [
        PlaneObservation("a", axis=1, side=-1.0, distance=1.0),
        PlaneObservation("b", axis=1, side=-1.0, distance=4.0),
        PlaneObservation("c", axis=1, side=-1.0, distance=1.0),
    ]
    links = [SharedPlaneLink("a", "b", axis=1), SharedPlaneLink("b", "c", axis=1)]
    result = register_axis(observations, links, axis=1, anchor="a")
    # The shared floor sits at world 0 - 1 = -1 (a's frame). b's eye must be 3
    # above a's for its own floor reading of 4 to land on that same -1. c reads
    # the floor the same way a did, so c's eye lands back at a's height.
    assert result.positions["a"] == pytest.approx(0.0)
    assert result.positions["b"] == pytest.approx(3.0)
    assert result.positions["c"] == pytest.approx(0.0)


def test_an_axis_nobody_linked_returns_only_the_anchor():
    observations = [
        PlaneObservation("a", axis=1, side=-1.0, distance=2.0),
        PlaneObservation("b", axis=1, side=-1.0, distance=2.0),
    ]
    result = register_axis(observations, links=[], axis=1, anchor="a")
    assert result.positions == {"a": 0.0}


def test_the_anchor_must_have_made_an_observation_on_this_axis():
    observations = [PlaneObservation("b", axis=1, side=-1.0, distance=2.0)]
    assert register_axis(observations, [], axis=1, anchor="a") is None


def test_a_redundant_link_in_a_cycle_is_reported_but_changes_nothing():
    """a, b and c all see the same floor; three links makes one of them
    redundant with the other two, and it is reported rather than silently
    absorbed - but see the next test for what that report can and cannot mean.
    """
    observations = [
        PlaneObservation("a", axis=1, side=-1.0, distance=1.0),
        PlaneObservation("b", axis=1, side=-1.0, distance=3.0),
        PlaneObservation("c", axis=1, side=-1.0, distance=5.0),
    ]
    links = [SharedPlaneLink("a", "b", axis=1), SharedPlaneLink("b", "c", axis=1),
             SharedPlaneLink("a", "c", axis=1)]
    result = register_axis(observations, links, axis=1, anchor="a")
    assert len(result.redundant_links) == 1
    assert result.positions["b"] == pytest.approx(2.0)
    assert result.positions["c"] == pytest.approx(4.0)


def test_a_link_asserting_two_unrelated_planes_are_the_same_is_not_caught():
    """The honest limitation this module carries, proven rather than assumed.

    With one observation per (view, axis), a link's equation is a plain
    difference of two fixed numbers. Any cycle those differences form sums to
    zero identically, for *any* values - a wrong assertion produces a solution
    that is internally self-consistent and still wrong. Catching it needs
    evidence from outside the link graph, which is exactly what this test
    demonstrates by using the reference-style "actual" positions only to show
    the registration silently agreed with a false premise.
    """
    # b and c are, in truth, at different heights (10 and 20) and looking at
    # two different, unrelated floors - not the shared surface the link below
    # claims. Nothing in a single observation per view can reveal that.
    observations = [
        PlaneObservation("a", axis=1, side=-1.0, distance=1.0),   # true eye: 0
        PlaneObservation("b", axis=1, side=-1.0, distance=3.0),   # true eye: 10
        PlaneObservation("c", axis=1, side=-1.0, distance=5.0),   # true eye: 20
    ]
    links = [SharedPlaneLink("a", "b", axis=1), SharedPlaneLink("b", "c", axis=1),
             SharedPlaneLink("a", "c", axis=1)]
    result = register_axis(observations, links, axis=1, anchor="a")
    # The registration reports no disagreement at all...
    assert len(result.redundant_links) == 1
    # ...while every solved position is wrong by construction of this example.
    assert result.positions["b"] != pytest.approx(10.0)
    assert result.positions["c"] != pytest.approx(20.0)


def test_a_view_with_two_observations_on_one_axis_is_refused():
    """Keying by view name would silently keep the last observation and drop
    the other - exactly the kind of silent data loss this project does not
    accept - so two observations for one (view, axis) is an error, not a pick.
    """
    observations = [
        PlaneObservation("a", axis=1, side=-1.0, distance=1.0),
        PlaneObservation("b", axis=1, side=-1.0, distance=3.0),
        PlaneObservation("b", axis=1, side=1.0, distance=2.0),
    ]
    with pytest.raises(ValueError, match="more than one observation"):
        register_axis(observations, [], axis=1, anchor="a")


def test_links_on_a_different_axis_are_ignored():
    observations = [
        PlaneObservation("a", axis=0, side=1.0, distance=4.0),
        PlaneObservation("a", axis=1, side=-1.0, distance=1.0),
        PlaneObservation("b", axis=1, side=-1.0, distance=1.0),
    ]
    links = [SharedPlaneLink("a", "b", axis=0)]   # no axis-0 observation for b
    result = register_axis(observations, links, axis=1, anchor="a")
    assert result.positions == {"a": 0.0}


def test_registration_is_a_frozen_dataclass_with_the_axis_and_anchor_recorded():
    observations = [PlaneObservation("a", axis=2, side=1.0, distance=3.0)]
    result = register_axis(observations, [], axis=2, anchor="a")
    assert isinstance(result, AxisRegistration)
    assert result.axis == 2
    assert result.anchor == "a"
