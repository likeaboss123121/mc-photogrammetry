"""Focused controls for the image-derived V2-B adapter."""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_image_derived_lattice_measure import _result as lattice_fixture, _producer1_record, _historical_bytes  # noqa: E402
from test_image_derived_endpoint_provenance import relative_result as relative_fixture  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_endpoint_provenance as o2  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_phase_anchor as subject  # noqa: E402
from mc_photogrammetry.reconstruction import plane_phase_anchor, plane_placement  # noqa: E402


@pytest.fixture(scope="module")
def records():
    patch = pytest.MonkeyPatch()
    try:
        _session, _pose, lattice = lattice_fixture(patch)
        relative = relative_fixture.__wrapped__()
        endpoint = o2.derive_image_derived_endpoint_provenance(
            relative_translations_result=relative)
        assert endpoint.status == "unavailable"
        yield endpoint, lattice
    finally:
        patch.undo()


def _forge(value, **changes):
    forged = object.__new__(type(value))
    for field in dataclasses.fields(value):
        object.__setattr__(forged, field.name,
                           changes.get(field.name, getattr(value, field.name)))
    return forged


def test_genuine_unavailable_chain_does_not_call_v2b(records, monkeypatch):
    endpoint, lattice = records
    monkeypatch.setattr(plane_phase_anchor, "analyze_phase_anchor",
                        lambda *_a, **_k: pytest.fail("V2-B must not run"))
    result = subject.derive_image_derived_phase_anchor(
        endpoint_provenance_result=endpoint, lattice_result=lattice)
    assert result.status == "unavailable"
    assert result.reason == "endpoint_provenance_unavailable"
    assert result.phase_anchor_result is None
    assert subject._hex(result.result_sha256)


def test_unavailable_repeat_is_deterministic(records):
    endpoint, lattice = records
    first = subject.derive_image_derived_phase_anchor(
        endpoint_provenance_result=endpoint, lattice_result=lattice)
    second = subject.derive_image_derived_phase_anchor(
        endpoint_provenance_result=endpoint, lattice_result=lattice)
    assert first == second


def test_producer1_measure_refuses_before_phase_numerics_and_producer3_binds(records, monkeypatch):
    endpoint, lattice = records
    current = subject.derive_image_derived_phase_anchor(
        endpoint_provenance_result=endpoint, lattice_result=lattice)
    assert lattice.producer == "m9c-image-derived-lattice-measure-3"
    assert current.lattice_result_sha256 == lattice.result_sha256
    historical = _producer1_record(lattice)
    before = _historical_bytes(historical)
    monkeypatch.setattr(plane_phase_anchor, "analyze_phase_anchor",
                        lambda *_a, **_k: pytest.fail("historical measure must refuse before phase analysis"))
    refused = subject.derive_image_derived_phase_anchor(
        endpoint_provenance_result=endpoint, lattice_result=historical)
    assert isinstance(refused, plane_placement.PlacementRefusal) and refused.reason == "provenance_mismatch"
    assert _historical_bytes(historical) == before


def test_stale_and_wrong_type_inputs_are_refused(records):
    endpoint, lattice = records
    stale = _forge(endpoint, result_sha256="0" * 64)
    assert isinstance(subject.derive_image_derived_phase_anchor(
        endpoint_provenance_result=stale, lattice_result=lattice),
        plane_placement.PlacementRefusal)
    assert isinstance(subject.derive_image_derived_phase_anchor(
        endpoint_provenance_result=endpoint, lattice_result=object()),
        plane_placement.PlacementRefusal)


def test_deep_v2b_replay_rejects_nested_forgery():
    from test_plane_phase_anchor import bundle, _anchor_budget, _run_anchor
    value = bundle.__wrapped__()
    result = _run_anchor(value, max_source_records=len(value["split_sources"]))
    assert result.status in ("partial", "determined")
    prediction = _forge(result.prediction, result_sha256="0" * 64)
    forged = _forge(result, prediction=prediction)
    with pytest.raises(ValueError):
        subject._replay_anchor(forged)


def test_public_api_has_no_manual_or_io_surface():
    import inspect
    assert tuple(inspect.signature(subject.derive_image_derived_phase_anchor).parameters) == (
        "endpoint_provenance_result", "lattice_result")
    source = Path(subject.__file__).read_text(encoding="utf-8").lower()
    for forbidden in ("manifest", "f3", "coordinate", "reference", "pathlib",
                      "open(", "subprocess", "argparse", "sys.argv", "environ",
                      "getenv", "config", "synthetic"):
        assert forbidden not in source
