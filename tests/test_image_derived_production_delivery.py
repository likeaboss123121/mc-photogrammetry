"""Focused controls for production-to-delivery composition."""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_image_derived_entry import _valid_result  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_delivery as delivery  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_production as production_mod  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_production_delivery as subject  # noqa: E402


def _production(monkeypatch, tmp_path):
    entry = _valid_result(monkeypatch, tmp_path)
    monkeypatch.setattr(production_mod.entry_mod, "build_image_derived_entry",
                        lambda *_a, **_k: entry)
    return production_mod.derive_image_output_from_paths(
        [tmp_path], project_path=tmp_path / "production.mcscan")


def test_nonpromoted_terminal_never_calls_delivery(monkeypatch, tmp_path):
    production = _production(monkeypatch, tmp_path)
    assert production.status != "promoted"
    monkeypatch.setattr(subject.production_mod, "derive_image_output_from_paths",
                        lambda *_a, **_k: production)
    monkeypatch.setattr(subject.output_delivery, "deliver_image_derived_output",
                        lambda **_k: pytest.fail("delivery must not run"))
    result = subject.derive_and_deliver_image_output(
        [tmp_path], destination=tmp_path / "x.litematic", name="x",
        description="x", project_path=tmp_path / "audit.mcscan")
    assert result.status == production.status
    assert result.production is production
    assert result.delivery is None and result.delivery_sha256 == ""


def test_promoted_output_is_delegated_once_with_exact_intent(monkeypatch,
                                                             tmp_path):
    promoted_output = object()
    production = SimpleNamespace(status="promoted", output=promoted_output)
    refusal = delivery._refuse("destination_exists", {"field": "test"})
    calls = []
    monkeypatch.setattr(subject.production_mod, "derive_image_output_from_paths",
                        lambda *_a, **_k: production)

    def deliver(**kwargs):
        calls.append(kwargs)
        return refusal

    monkeypatch.setattr(subject.output_delivery,
                        "deliver_image_derived_output", deliver)
    monkeypatch.setattr(subject, "_make",
                        lambda prod, **kwargs: (prod, kwargs))
    destination = tmp_path / "exact.litematic"
    result = subject.derive_and_deliver_image_output(
        [Path("a.png")], destination=destination, name="Exact",
        description="Bound", overwrite=True,
        project_path=tmp_path / "audit.mcscan")
    assert calls == [{
        "output_result": promoted_output, "destination": destination,
        "name": "Exact", "description": "Bound", "overwrite": True,
        "exporter": None}]
    assert result == (production, {
        "status": "refused", "reason": "destination_exists",
        "delivery": refusal})


def test_public_surface_has_only_delivery_intent_and_m7_caps():
    import inspect
    names = tuple(inspect.signature(
        subject.derive_and_deliver_image_output).parameters)
    assert names == (
        "image_inputs", "destination", "name", "description", "overwrite",
        "project_path", "recursive", "max_frames", "max_pairs",
        "max_total_support", "max_edges", "max_cycle_checks",
        "rotation_tolerance_degrees", "direction_tolerance")
    source = Path(subject.__file__).read_text(encoding="utf-8").lower()
    for forbidden in ("manual_metadata", "palette", "manifest", "f3",
                      "coordinate", "bounds", "camera", "scene"):
        assert forbidden not in source
