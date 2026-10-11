"""Focused controls for the image-derived placement assembly seam.

The seam under test is the boundary between the accepted DS-M9-C lattice
measurement and the accepted M4 ``PlacementUniverse`` assembler.  These controls
are deliberately hostile about scope: the seam must propagate a placement
refusal verbatim, propagate an unavailable lattice as one typed unavailable,
reject a foreign or malformed lattice without duck-typing, refuse an absent or
non-conforming bundle as ``missing_prerequisite``, and delegate an exact,
complete bundle to the accepted assembler unchanged.

The acceptance, delegation and rejection controls drive the real accepted image
chain (through the measurement seam's own focused fixture) so the sealed
assembled lattice is a genuine producer record whose readings and bridge are
bound to the bundle under test; the hostile controls additionally forge a
constructor-bypassed lattice and swap bundle entries rather than trusting the
seam's instance check alone.
"""
from __future__ import annotations

import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_image_derived_lattice_measure import (  # noqa: E402
    _build_session,
    _pose_graph,
    _result,
)

from mc_photogrammetry.pipeline import image_derived_lattice_measure as m9c  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_placement as seam  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_session as session_mod  # noqa: E402
from mc_photogrammetry.reconstruction import plane_placement as placement  # noqa: E402
from mc_photogrammetry.reconstruction import plane_placement_assembly as assembly  # noqa: E402


def _assembled(**changes):
    """A constructor-bypassing stand-in for an accepted assembled lattice result.

    It never runs ``__post_init__``, so the seam's own declared-field replay - not
    this fixture - is what must reject it.
    """
    lattice = object.__new__(m9c.ImageDerivedLatticeResult)
    defaults = {
        "terminal": m9c.GEOMETRY_UNAVAILABLE,
        "status": "assembled",
        "reason": "",
        "readings": (object(),),
        "bridge": object(),
    }
    defaults.update(changes)
    for name, value in defaults.items():
        object.__setattr__(lattice, name, value)
    return lattice


def _full_bundle():
    return {name: object() for name in seam.REQUIRED_PLACEMENT_KEYWORDS}


def _genuine_bundle(lattice):
    """A complete bundle whose readings and split sources are the lattice's own."""
    bundle = _full_bundle()
    bundle["readings"] = lattice.readings
    bundle["split_sources"] = lattice.bridge.split_sources_for_assembly()
    return bundle


def _unavailable_lattice(monkeypatch):
    """A genuine typed-unavailable terminal from the real accepted measurement."""
    session = _build_session(monkeypatch)
    pose_graph = _pose_graph(session)
    stripped = session_mod.image_derived_session_from_record(
        session_mod.image_derived_session_to_record(session))
    lattice = m9c.measure_image_derived_lattice(
        session=stripped, pose_graph=pose_graph)
    assert isinstance(lattice, m9c.ImageDerivedLatticeResult)
    assert lattice.status == "unavailable"
    return lattice


# ---------------------------------------------------------------------------
# The accepted signature and the unavailable record
# ---------------------------------------------------------------------------

def test_the_required_keywords_are_the_accepted_assembler_signature():
    assert seam.REQUIRED_PLACEMENT_KEYWORDS == (
        "axis_support", "p1d_result", "vector_result", "c2rb_result",
        "v2a_result", "v2b_result", "readings", "signatures", "matches",
        "phase_evidence", "usable_masks", "split_sources", "anchor_budget",
        "support_budget", "comparison_tolerance", "closure_tolerance")


def test_the_unavailable_record_is_frozen_and_closed():
    record = seam.ImageDerivedPlacementUnavailable(
        terminal="geometry_unavailable", reason="missing_prerequisite", detail="d")
    assert record.terminal == seam.PLACEMENT_UNAVAILABLE_TERMINAL
    with pytest.raises(FrozenInstanceError):
        record.terminal = "other"
    with pytest.raises(ValueError):
        seam.ImageDerivedPlacementUnavailable(
            terminal="other", reason="geometry_unavailable", detail="d")
    with pytest.raises(ValueError):
        seam.ImageDerivedPlacementUnavailable(
            terminal="geometry_unavailable", reason="other", detail="d")
    with pytest.raises(ValueError):
        seam.ImageDerivedPlacementUnavailable(
            terminal="geometry_unavailable", reason="geometry_unavailable", detail=1)


# ---------------------------------------------------------------------------
# Lattice propagation and rejection
# ---------------------------------------------------------------------------

def test_a_lattice_placement_refusal_is_propagated_verbatim():
    refusal = placement._make_refusal("invalid_input", {"field": "session"})
    result = seam.assemble_image_derived_placement(lattice_result=refusal)
    assert result is refusal


def test_a_genuine_unavailable_lattice_is_propagated_as_typed_unavailable(monkeypatch):
    lattice = _unavailable_lattice(monkeypatch)
    result = seam.assemble_image_derived_placement(
        lattice_result=lattice, placement_records=_full_bundle())
    assert isinstance(result, seam.ImageDerivedPlacementUnavailable)
    assert result.terminal == "geometry_unavailable"
    assert result.reason == "geometry_unavailable"
    assert "image_unavailable" in result.detail


def test_a_foreign_lattice_result_is_rejected_without_duck_typing():
    class Foreign:
        terminal = m9c.GEOMETRY_UNAVAILABLE
        status = "assembled"
        reason = ""
        readings = (object(),)
        bridge = object()

    result = seam.assemble_image_derived_placement(
        lattice_result=Foreign(), placement_records=_full_bundle())
    assert isinstance(result, seam.ImageDerivedPlacementUnavailable)
    assert result.terminal == "geometry_unavailable"
    assert result.reason == "geometry_unavailable"


def test_a_malformed_assembled_lattice_is_rejected():
    changes = ({"terminal": "other"}, {"status": "other"},
               {"readings": ()}, {"bridge": None})
    for change in changes:
        result = seam.assemble_image_derived_placement(
            lattice_result=_assembled(**change), placement_records=_full_bundle())
        assert isinstance(result, seam.ImageDerivedPlacementUnavailable), change
        assert result.terminal == "geometry_unavailable"
        assert result.reason == "geometry_unavailable"


def test_a_constructor_bypassed_lattice_with_fake_readings_and_bridge_is_rejected():
    lattice = _assembled(readings=(object(),), bridge=object())
    result = seam.assemble_image_derived_placement(
        lattice_result=lattice, placement_records=_full_bundle())
    assert isinstance(result, seam.ImageDerivedPlacementUnavailable)
    assert result.terminal == "geometry_unavailable"
    assert result.reason == "geometry_unavailable"


def test_a_constructor_bypassed_unavailable_lattice_is_not_trusted():
    lattice = _assembled(status="unavailable", reason="image_unavailable",
                         readings=(), bridge=None)
    result = seam.assemble_image_derived_placement(
        lattice_result=lattice, placement_records=_full_bundle())
    assert isinstance(result, seam.ImageDerivedPlacementUnavailable)
    assert result.reason == "geometry_unavailable"
    assert "image_unavailable" not in result.detail


# ---------------------------------------------------------------------------
# Bundle discipline
# ---------------------------------------------------------------------------

def test_a_genuine_assembled_lattice_without_a_bundle_needs_a_prerequisite(monkeypatch):
    _session, _pose, lattice = _result(monkeypatch)
    assert lattice.status == "assembled" and lattice.bridge is not None
    result = seam.assemble_image_derived_placement(lattice_result=lattice)
    assert isinstance(result, seam.ImageDerivedPlacementUnavailable)
    assert result.terminal == "geometry_unavailable"
    assert result.reason == "missing_prerequisite"
    assert "axis_support" in result.detail
    assert "closure_tolerance" in result.detail


def test_an_incomplete_bundle_names_the_missing_keyword(monkeypatch):
    _session, _pose, lattice = _result(monkeypatch)
    bundle = _genuine_bundle(lattice)
    del bundle["readings"]
    result = seam.assemble_image_derived_placement(
        lattice_result=lattice, placement_records=bundle)
    assert result.reason == "missing_prerequisite"
    assert "readings" in result.detail


def test_an_extra_bundle_key_names_the_unexpected_keyword(monkeypatch):
    _session, _pose, lattice = _result(monkeypatch)
    bundle = _genuine_bundle(lattice)
    bundle["fabricated"] = object()
    result = seam.assemble_image_derived_placement(
        lattice_result=lattice, placement_records=bundle)
    assert result.reason == "missing_prerequisite"
    assert "fabricated" in result.detail


def test_a_non_mapping_bundle_is_missing_prerequisite(monkeypatch):
    _session, _pose, lattice = _result(monkeypatch)
    result = seam.assemble_image_derived_placement(
        lattice_result=lattice, placement_records=list(_genuine_bundle(lattice)))
    assert result.reason == "missing_prerequisite"


# ---------------------------------------------------------------------------
# Swapped-bundle binding
# ---------------------------------------------------------------------------

def test_a_reordered_readings_bundle_is_rejected(monkeypatch):
    _session, _pose, lattice = _result(monkeypatch)
    bundle = _genuine_bundle(lattice)
    bundle["readings"] = tuple(reversed(lattice.readings))
    result = seam.assemble_image_derived_placement(
        lattice_result=lattice, placement_records=bundle)
    assert isinstance(result, seam.ImageDerivedPlacementUnavailable)
    assert result.terminal == "geometry_unavailable"
    assert result.reason == "geometry_unavailable"


def test_a_content_equal_but_distinct_readings_bundle_is_rejected(monkeypatch):
    _session, _pose, lattice = _result(monkeypatch)
    bundle = _genuine_bundle(lattice)
    bundle["readings"] = tuple(replace(reading) for reading in lattice.readings)
    result = seam.assemble_image_derived_placement(
        lattice_result=lattice, placement_records=bundle)
    assert isinstance(result, seam.ImageDerivedPlacementUnavailable)
    assert result.reason == "geometry_unavailable"


def test_a_reordered_split_sources_bundle_is_rejected(monkeypatch):
    _session, _pose, lattice = _result(monkeypatch)
    bundle = _genuine_bundle(lattice)
    bundle["split_sources"] = tuple(
        reversed(lattice.bridge.split_sources_for_assembly()))
    result = seam.assemble_image_derived_placement(
        lattice_result=lattice, placement_records=bundle)
    assert isinstance(result, seam.ImageDerivedPlacementUnavailable)
    assert result.reason == "geometry_unavailable"


def test_a_tampered_split_source_patch_is_rejected(monkeypatch):
    _session, _pose, lattice = _result(monkeypatch)
    bundle = _genuine_bundle(lattice)
    source_result, patch, budget = bundle["split_sources"][0]
    bundle["split_sources"] = (
        (source_result, patch + 1.0, budget),) + bundle["split_sources"][1:]
    result = seam.assemble_image_derived_placement(
        lattice_result=lattice, placement_records=bundle)
    assert isinstance(result, seam.ImageDerivedPlacementUnavailable)
    assert result.reason == "geometry_unavailable"


# ---------------------------------------------------------------------------
# Unchanged delegation
# ---------------------------------------------------------------------------

def test_the_exact_bundle_is_delegated_unchanged(monkeypatch):
    _session, _pose, lattice = _result(monkeypatch)
    sentinel = object()
    seen = {}

    def fake_assemble(**kwargs):
        seen.update(kwargs)
        return sentinel

    monkeypatch.setattr(assembly, "assemble_placement_universe", fake_assemble)
    bundle = _genuine_bundle(lattice)
    result = seam.assemble_image_derived_placement(
        lattice_result=lattice, placement_records=bundle)
    assert result is sentinel
    assert set(seen) == set(seam.REQUIRED_PLACEMENT_KEYWORDS)
    assert seen["readings"] is lattice.readings
    assert all(seen[name] is bundle[name] for name in bundle)


def test_an_assembler_refusal_is_returned_unchanged(monkeypatch):
    refusal = placement._make_refusal("invalid_input", {"field": "axis_support"})

    def fake_assemble(**_kwargs):
        return refusal

    monkeypatch.setattr(assembly, "assemble_placement_universe", fake_assemble)
    _session, _pose, lattice = _result(monkeypatch)
    result = seam.assemble_image_derived_placement(
        lattice_result=lattice, placement_records=_genuine_bundle(lattice))
    assert result is refusal
