"""Focused controls for the image-derived production-output adapter."""
from __future__ import annotations

import dataclasses
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_image_derived_lattice_measure import _result as lattice_fixture  # noqa: E402
from test_image_derived_orchestration import recipe as accepted_recipe  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_geometry_orchestration as r1  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_output as subject  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_output_policy as policy_mod  # noqa: E402
from mc_photogrammetry.reconstruction import plane_placement  # noqa: E402


@pytest.fixture(scope="module")
def records():
    patch = pytest.MonkeyPatch()
    try:
        session, _pose, lattice = lattice_fixture(patch)
        geometry = r1.derive_image_derived_geometry(
            session=session, lattice_result=lattice)
        assert geometry.status == "unavailable"
        yield session, geometry
    finally:
        patch.undo()


def _forge(value, **changes):
    forged = object.__new__(type(value))
    for field in dataclasses.fields(value):
        object.__setattr__(forged, field.name,
                           changes.get(field.name, getattr(value, field.name)))
    return forged


def test_current_chain_stops_before_output_producers(records, monkeypatch):
    session, geometry = records
    monkeypatch.setattr(subject.placement_mod, "resolve_face_placement",
                        lambda **_kwargs: pytest.fail("placement must not run"))
    result = subject.derive_image_derived_output(
        geometry_result=geometry, session=session)
    assert result.status == "unavailable"
    assert result.reason == "geometry_unavailable"
    assert result.geometry_sha256 == geometry.result_sha256
    assert result.policy_sha256 == policy_mod.IMAGE_DERIVED_OUTPUT_POLICY.policy_sha256
    assert result.placement is None and result.volume is None
    assert not result.frames and result.promotion is None


def test_unavailable_result_is_repeatable(records):
    session, geometry = records
    first = subject.derive_image_derived_output(
        geometry_result=geometry, session=session)
    second = subject.derive_image_derived_output(
        geometry_result=geometry, session=session)
    assert first == second


def test_tampered_and_foreign_geometry_are_refused(records):
    session, geometry = records
    stale = _forge(geometry, result_sha256="0" * 64)
    assert isinstance(subject.derive_image_derived_output(
        geometry_result=stale, session=session),
        plane_placement.PlacementRefusal)
    assert isinstance(subject.derive_image_derived_output(
        geometry_result=object(), session=session),
        plane_placement.PlacementRefusal)


def test_wrapper_digest_and_policy_binding(records):
    session, geometry = records
    result = subject.derive_image_derived_output(
        geometry_result=geometry, session=session)
    with pytest.raises(ValueError, match="digest mismatch"):
        dataclasses.replace(result, result_sha256="0" * 64)
    forged_policy = _forge(result.policy, policy_sha256="0" * 64)
    with pytest.raises(ValueError):
        dataclasses.replace(result, policy=forged_policy)


def test_accepted_recipe_delegates_with_exact_identities(monkeypatch):
    recipe = accepted_recipe.__wrapped__()
    contexts = recipe["frames"]
    live_frames = tuple(SimpleNamespace(
        frame_id=item.frame_id, image=item.image,
        image_sha256=item.image_sha256) for item in contexts)
    masks = tuple(SimpleNamespace(frame_id=item.frame_id,
                                  mask=item.usable_mask) for item in contexts)
    lattice = SimpleNamespace(readings=recipe["universe"].readings)
    geometry = SimpleNamespace(
        universe=recipe["universe"], supports=recipe["supports"],
        records=(("f1", SimpleNamespace(usable_masks=masks)),
                 ("p2", SimpleNamespace(lattice_result=lattice))))
    seen = []
    original_place = subject.placement_mod.resolve_face_placement
    original_volume = subject.volume_mod.build_plane_volume

    def place(**kwargs):
        seen.append(("placement", kwargs))
        return original_place(**kwargs)

    def volume(**kwargs):
        seen.append(("volume", kwargs))
        return original_volume(**kwargs)

    monkeypatch.setattr(subject.placement_mod, "resolve_face_placement", place)
    monkeypatch.setattr(subject.volume_mod, "build_plane_volume", volume)
    values = subject._derive_ready(
        geometry=geometry, live_frames=live_frames,
        policy=policy_mod.IMAGE_DERIVED_OUTPUT_POLICY)
    assert seen[0][0] == "placement"
    assert seen[0][1]["universe"] is recipe["universe"]
    assert seen[0][1]["supports"] is recipe["supports"]
    assert seen[0][1]["budget"] is policy_mod.IMAGE_DERIVED_OUTPUT_POLICY.placement_budget
    assert seen[1][0] == "volume"
    assert seen[1][1]["placement"] is values[2]
    assert seen[1][1]["budget"] is policy_mod.IMAGE_DERIVED_OUTPUT_POLICY.volume_budget
    assert values[0] in ("promoted", "review", "refused", "budget_exhausted")


def test_public_surface_has_no_manual_or_io_inputs():
    assert tuple(inspect.signature(subject.derive_image_derived_output).parameters) == (
        "geometry_result", "session")
    source = Path(subject.__file__).read_text(encoding="utf-8").lower()
    for forbidden in ("manifest", "manual_metadata", "pathlib", "open(",
                      "subprocess", "argparse", "sys.argv", "environ",
                      "getenv", "synthetic", "destination", "exporter"):
        assert forbidden not in source
    assert len(source.splitlines()) < 750
