"""Focused controls for the R2-A verified-delivery gate."""
from __future__ import annotations

import dataclasses
import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_image_derived_delivery import happy_promotion  # noqa: E402
from test_image_derived_lattice_measure import _result as lattice_fixture  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_delivery as delivery  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_geometry_orchestration as r1  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_output as r2  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_output_delivery as subject  # noqa: E402


@pytest.fixture(scope="module")
def unavailable_output():
    patch = pytest.MonkeyPatch()
    try:
        session, _pose, lattice = lattice_fixture(patch)
        geometry = r1.derive_image_derived_geometry(
            session=session, lattice_result=lattice)
        result = r2.derive_image_derived_output(
            geometry_result=geometry, session=session)
        assert result.status == "unavailable"
        yield result
    finally:
        patch.undo()


def _forge(value, **changes):
    forged = object.__new__(type(value))
    for field in dataclasses.fields(value):
        object.__setattr__(forged, field.name,
                           changes.get(field.name, getattr(value, field.name)))
    return forged


def test_nonpromoted_output_refuses_before_delivery(unavailable_output,
                                                     monkeypatch, tmp_path):
    monkeypatch.setattr(delivery, "deliver_image_derived_litematic",
                        lambda *_a, **_k: pytest.fail("delivery must not run"))
    result = subject.deliver_image_derived_output(
        output_result=unavailable_output, destination=tmp_path / "x.litematic",
        name="x", description="x")
    assert isinstance(result, delivery.DeliveryRefusal)
    assert result.reason == "invalid_promotion"


def test_tampered_and_foreign_outputs_are_typed_refusals(unavailable_output,
                                                          tmp_path):
    stale = _forge(unavailable_output, result_sha256="0" * 64)
    for value in (stale, object()):
        result = subject.deliver_image_derived_output(
            output_result=value, destination=tmp_path / "x.litematic",
            name="x", description="x")
        assert isinstance(result, delivery.DeliveryRefusal)
        assert result.reason == "invalid_input"


def test_exact_delegation_once(monkeypatch, tmp_path):
    promoted = happy_promotion.__wrapped__()
    expected = delivery._refuse("destination_exists", {"field": "test"})
    calls = []
    exporter = object()

    def delegated(value, destination, **kwargs):
        calls.append((value, destination, kwargs))
        return expected

    monkeypatch.setattr(delivery, "deliver_image_derived_litematic", delegated)
    destination = tmp_path / "exact.litematic"
    result = subject._deliver_promoted(
        promoted, destination=destination, name="Exact",
        description="Bound", overwrite=True, exporter=exporter)
    assert result is expected
    assert calls == [(promoted, destination, {
        "name": "Exact", "description": "Bound", "overwrite": True,
        "exporter": exporter})]


@pytest.mark.parametrize("field,value,reason", [
    ("export_ready", False, "promotion_not_export_ready"),
    ("exact_state_eligible", False, "promotion_not_exact_state_eligible"),
    ("omitted_unknown", 1, "promotion_omitted_unknown"),
])
def test_eligibility_gates_precede_delegation(field, value, reason,
                                               monkeypatch, tmp_path):
    promoted = _forge(happy_promotion.__wrapped__(), **{field: value})
    monkeypatch.setattr(delivery, "deliver_image_derived_litematic",
                        lambda *_a, **_k: pytest.fail("delivery must not run"))
    result = subject._deliver_promoted(
        promoted, destination=tmp_path / "x.litematic", name="x",
        description="x", overwrite=False, exporter=None)
    assert isinstance(result, delivery.DeliveryRefusal)
    assert result.reason == reason


def test_public_signature_and_no_reconstruction_surface():
    signature = inspect.signature(subject.deliver_image_derived_output)
    assert tuple(signature.parameters) == (
        "output_result", "destination", "name", "description", "overwrite",
        "exporter")
    source = Path(subject.__file__).read_text(encoding="utf-8").lower()
    for forbidden in ("manifest", "manual_metadata", "placement", "volume",
                      "face_observation", "propose_face_states",
                      "promote_plane_states", "tempfile", "os.replace", "open("):
        assert forbidden not in source
    assert len(source.splitlines()) < 300
