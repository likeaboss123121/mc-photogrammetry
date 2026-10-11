"""Focused controls for the image-derived V1 support adapter."""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_image_derived_lattice_measure import _result as lattice_fixture  # noqa: E402
from test_image_derived_endpoint_provenance import relative_result as relative_fixture  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_endpoint_provenance as o2  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_phase_anchor as p2  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_region_support as subject  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_signatures as f1  # noqa: E402
from mc_photogrammetry.reconstruction import plane_placement, plane_region_support  # noqa: E402


@pytest.fixture(scope="module")
def records():
    patch = pytest.MonkeyPatch()
    try:
        session, _pose, lattice = lattice_fixture(patch)
        signatures = f1.derive_image_derived_signatures(
            session=session, lattice_result=lattice)
        relative = relative_fixture.__wrapped__()
        endpoint = o2.derive_image_derived_endpoint_provenance(
            relative_translations_result=relative)
        anchor = p2.derive_image_derived_phase_anchor(
            endpoint_provenance_result=endpoint, lattice_result=lattice)
        assert anchor.status == "unavailable" and signatures.status == "assembled"
        yield session, signatures, anchor
    finally:
        patch.undo()


def _forge(value, **changes):
    forged = object.__new__(type(value))
    for field in dataclasses.fields(value):
        object.__setattr__(forged, field.name,
                           changes.get(field.name, getattr(value, field.name)))
    return forged


def test_genuine_unavailable_chain_does_not_call_v1(records, monkeypatch):
    session, signatures, anchor = records
    monkeypatch.setattr(plane_region_support, "build_aligned_region_support",
                        lambda **_k: pytest.fail("V1 must not run"))
    result = subject.derive_image_derived_region_support(
        phase_anchor_result=anchor, signatures_result=signatures,
        session=session)
    assert result.status == "unavailable"
    assert result.reason == "phase_anchor_unavailable"
    assert not result.support_evidence and not result.refusals


def test_repeat_is_deterministic(records):
    session, signatures, anchor = records
    first = subject.derive_image_derived_region_support(
        phase_anchor_result=anchor, signatures_result=signatures, session=session)
    second = subject.derive_image_derived_region_support(
        phase_anchor_result=anchor, signatures_result=signatures, session=session)
    assert first == second


def test_stale_and_wrong_inputs_are_refused(records):
    session, signatures, anchor = records
    stale = _forge(anchor, result_sha256="0" * 64)
    assert isinstance(subject.derive_image_derived_region_support(
        phase_anchor_result=stale, signatures_result=signatures,
        session=session), plane_placement.PlacementRefusal)
    assert isinstance(subject.derive_image_derived_region_support(
        phase_anchor_result=anchor, signatures_result=object(), session=session),
        plane_placement.PlacementRefusal)


def test_deep_support_and_refusal_replay():
    from test_plane_region_support import _rect_support, _build, _identity
    reading, support = _rect_support()
    evidence = subject.RegionSupportEvidence(
        support.reading_identity_sha256, "1" * 64,
        support.usable_mask_sha256, support.image_sha256, support)
    assert subject._deep_replay(evidence) == evidence
    refusal = _build(reading, _identity(reading),
                     __import__("numpy").ones((4, 4), dtype=bool),
                     grid_shape=(10_000_001, 1))
    assert isinstance(refusal, plane_region_support.AlignedSupportRefusal)
    assert subject._deep_replay(refusal) == refusal


def test_public_api_has_no_manual_or_io_surface():
    import inspect
    assert tuple(inspect.signature(subject.derive_image_derived_region_support).parameters) == (
        "phase_anchor_result", "signatures_result", "session")
    source = Path(subject.__file__).read_text(encoding="utf-8").lower()
    for forbidden in ("manifest", "f3", "coordinate", "reference", "pathlib",
                      "open(", "subprocess", "argparse", "sys.argv", "environ",
                      "getenv", "config", "synthetic"):
        assert forbidden not in source
