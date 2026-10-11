"""Focused controls for image-derived geometry orchestration."""
from __future__ import annotations

import dataclasses
import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_image_derived_lattice_measure import _result as lattice_fixture, _producer1_record, _historical_bytes  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_geometry_orchestration as subject  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_axis_support as h_mod  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_graph as g_mod  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_graph_policy as j_mod  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_session as session_mod  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_signatures as f1_mod  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_shift_domains as k_mod  # noqa: E402
from mc_photogrammetry.reconstruction import plane_placement  # noqa: E402


@pytest.fixture(scope="module")
def records():
    patch = pytest.MonkeyPatch()
    try:
        session, _pose, lattice = lattice_fixture(patch)
        yield session, lattice
    finally:
        patch.undo()


def _forge(value, **changes):
    forged = object.__new__(type(value))
    for field in dataclasses.fields(value):
        object.__setattr__(forged, field.name,
                           changes.get(field.name, getattr(value, field.name)))
    return forged


def test_current_fixture_stops_at_first_honest_status(records, monkeypatch):
    session, lattice = records
    monkeypatch.setattr(h_mod, "derive_image_derived_axis_support",
                        lambda **_kwargs: pytest.fail("H must not run after G"))
    monkeypatch.setattr(j_mod, "bind_image_derived_graph_policy",
                        lambda **_kwargs: pytest.fail("J must not run after H"))
    result = subject.derive_image_derived_geometry(
        session=session, lattice_result=lattice)
    assert result.status == "unavailable"
    assert result.reason == "g_unavailable"
    assert tuple(name for name, _ in result.records) == ("f1", "g")
    assert result.records[0][1].status == "assembled"
    assert result.records[0][1].usable_masks
    assert not result.placement_records and result.universe is None


def test_repeat_is_digest_stable(records):
    session, lattice = records
    first = subject.derive_image_derived_geometry(
        session=session, lattice_result=lattice)
    second = subject.derive_image_derived_geometry(
        session=session, lattice_result=lattice)
    assert first.result_sha256 == second.result_sha256
    assert first.stage_digests == second.stage_digests


def test_producer1_measure_refuses_before_geometry_stages_and_producer3_binds(records, monkeypatch):
    session, lattice = records
    current = subject.derive_image_derived_geometry(session=session, lattice_result=lattice)
    assert lattice.producer == "m9c-image-derived-lattice-measure-3"
    assert current.lattice_sha256 == lattice.result_sha256
    assert current.records[0][1].lattice_sha256 == lattice.result_sha256
    historical = _producer1_record(lattice)
    before = _historical_bytes(historical)
    monkeypatch.setattr(f1_mod, "derive_image_derived_signatures",
                        lambda **_kw: pytest.fail("historical measure must refuse before F1"))
    refused = subject.derive_image_derived_geometry(session=session, lattice_result=historical)
    assert isinstance(refused, plane_placement.PlacementRefusal) and refused.reason == "provenance_mismatch"
    assert _historical_bytes(historical) == before


def test_exact_connection_graph_is_forwarded_to_g(records,monkeypatch):
    from test_image_derived_graph import _current_sources
    source=_current_sources(monkeypatch)
    monkeypatch.setattr(subject.f1_mod,"derive_image_derived_signatures",lambda **kwargs:source["signatures_result"])
    original=subject.g_mod.derive_image_derived_graph;seen=[]
    def derive(**kwargs):
        seen.append(kwargs["connection_graph"]);return original(**kwargs)
    monkeypatch.setattr(subject.g_mod,"derive_image_derived_graph",derive)
    result=subject.derive_image_derived_geometry(session=source["session"],lattice_result=source["lattice_result"],connection_graph=source["connection_graph"])
    assert seen == [source["connection_graph"]]
    assert dict(result.records)["g"].status == "assembled"


def test_validated_live_session_reaches_f1_with_pixels(records, monkeypatch):
    session, lattice = records
    original = f1_mod.derive_image_derived_signatures
    seen = []

    def derive(**kwargs):
        forwarded = kwargs["session"]
        seen.append(forwarded)
        assert forwarded is session
        assert all(frame.image is not None for frame in forwarded.frames)
        return original(**kwargs)

    monkeypatch.setattr(f1_mod, "derive_image_derived_signatures", derive)
    result = subject.derive_image_derived_geometry(
        session=session, lattice_result=lattice)
    assert seen == [session]
    assert result.records[0][0] == "f1"
    assert result.records[0][1].status == "assembled"


def test_pixel_free_session_is_typed_image_unavailable(records):
    session, lattice = records
    pixel_free = session_mod.image_derived_session_from_record(
        session_mod.image_derived_session_to_record(session))
    result = subject.derive_image_derived_geometry(
        session=pixel_free, lattice_result=lattice)
    assert result.status == "unavailable"
    assert result.reason == "f1_unavailable"
    assert result.records[0][1].reason == "image_unavailable"


def test_tampered_live_session_is_refused(records):
    session, lattice = records
    bad_frame = _forge(session.frames[0], image_sha256="0" * 64)
    tampered = _forge(session, frames=(bad_frame,))
    result = subject.derive_image_derived_geometry(
        session=tampered, lattice_result=lattice)
    assert isinstance(result, plane_placement.PlacementRefusal)


def test_tampered_lattice_and_foreign_inputs_are_refused(records):
    session, lattice = records
    stale = _forge(lattice, result_sha256="0" * 64)
    assert isinstance(subject.derive_image_derived_geometry(
        session=session, lattice_result=stale), plane_placement.PlacementRefusal)
    assert isinstance(subject.derive_image_derived_geometry(
        session=session, lattice_result=object()),
        plane_placement.PlacementRefusal)
    assert isinstance(subject.derive_image_derived_geometry(
        session=object(), lattice_result=lattice),
        plane_placement.PlacementRefusal)


def test_wrapper_rejects_stage_digest_tampering(records):
    session, lattice = records
    result = subject.derive_image_derived_geometry(
        session=session, lattice_result=lattice)
    with pytest.raises(ValueError, match="stage digests"):
        dataclasses.replace(result, stage_digests=tuple(
            (name, "0" * 64) for name, _digest in result.stage_digests))
    with pytest.raises(ValueError, match="canonical prefix"):
        dataclasses.replace(result, records=(("g", result.records[0][1]),))


def test_public_api_and_source_surface():
    assert tuple(inspect.signature(
        subject.derive_image_derived_geometry).parameters) == (
            "session", "lattice_result", "connection_graph")
    source = Path(subject.__file__).read_text(encoding="utf-8").lower()
    for forbidden in ("manifest", "manual_metadata", "pathlib", "open(",
                      "subprocess", "argparse", "sys.argv", "environ",
                      "getenv", "synthetic", "**kwargs"):
        assert forbidden not in source
    assert len(source.splitlines()) < 600


def test_exact_placement_key_order_is_the_accepted_signature():
    assert subject.d_mod.REQUIRED_PLACEMENT_KEYWORDS == (
        "axis_support", "p1d_result", "vector_result", "c2rb_result",
        "v2a_result", "v2b_result", "readings", "signatures", "matches",
        "phase_evidence", "usable_masks", "split_sources", "anchor_budget",
        "support_budget", "comparison_tolerance", "closure_tolerance")
