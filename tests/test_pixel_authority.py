"""One usable-pixel authority per reconstruction call.

DS-4 R3.  ``reconstruct_scene_surfaces`` normalises the usable-pixel domain
exactly once and every pixel consumer reads that one map:

* face sampling receives its inverse as the GUI/exclusion mask;
* the residual proposal pass receives the usable map;
* partial shape fitting receives the same usable map.

A caller-supplied ``usable_masks`` mapping is authoritative - a missing frame is
all-false and a wrong-shaped mask raises before anything is sampled or scored -
and ``None`` derives the authority from ``~create_gui_mask`` so the thin-block
path no longer samples under the HUD and then scores residuals without it.

The controls below are spies, not copies of the implementation: they record the
exact objects each consumer was handed and never reach the real scoring code.
"""
import numpy as np
import pytest

from mc_photogrammetry.domain.models import BlockState
from mc_photogrammetry.reconstruction import raycast
from mc_photogrammetry.reconstruction import residual
from mc_photogrammetry.reconstruction import shape_fit
from mc_photogrammetry.reconstruction.camera import CameraModel
from mc_photogrammetry.reconstruction.facade import PaletteColor
from mc_photogrammetry.reconstruction.scene import SceneTransform
from mc_photogrammetry.vision.gui_mask import create_gui_mask

SHAPE_A = (100, 80)
SHAPE_B = (60, 70)


def _frame(shape):
    return np.zeros((shape[0], shape[1], 3), dtype=np.uint8)


def _camera():
    return CameraModel(position=(2.5, 4.0, -4.0), yaw_degrees=0.0,
                       pitch_degrees=40.0, fov_degrees=90.0)


def _palette():
    return (PaletteColor(BlockState("minecraft:stone"), (125, 125, 125)),)


def _scene():
    return SceneTransform(world_origin=(0, 0, 0), size=(6, 6, 6))


def _install_spies(monkeypatch):
    """Spy the three pixel consumers and short-circuit them.

    ``_surface_of`` is pinned to the empty set so ``extra_proposals`` always
    reaches the second sampling pass; that is the pass the contract says must
    share the authority with the first.
    """
    captured = {"sampling": [], "residual": None, "fitting": None}

    def sampling_spy(coordinates, camera, scene, image, **kwargs):
        captured["sampling"].append((image, kwargs.get("gui_mask")))
        return []

    def residual_spy(blocks, scene, cameras, frames, palette, **kwargs):
        captured["residual"] = kwargs.get("usable_masks")
        return {}

    def fitting_spy(cells, blocks, scene, cameras, frames, palette, **kwargs):
        captured["fitting"] = kwargs.get("usable_masks")
        return {}

    monkeypatch.setattr(raycast, "sample_visible_faces", sampling_spy)
    monkeypatch.setattr(residual, "residual_proposals", residual_spy)
    monkeypatch.setattr(shape_fit, "fit_states", fitting_spy)
    monkeypatch.setattr(raycast, "_surface_of", lambda coordinates: set())
    return captured


def _reconstruct(frames, *, usable_masks, fit_partial_shapes=True):
    return raycast.reconstruct_scene_surfaces(
        frames, {name: _camera() for name in frames}, _scene(),
        palette=_palette(),
        propose_colour_gap=0.0,
        adjacency_strength=0.0,
        simulate_lighting=False,
        correct_light_cast=False,
        reconcile_exposure=False,
        self_calibrate=False,
        surface_only=False,
        fit_partial_shapes=fit_partial_shapes,
        extra_proposals={(0, 0, 0)},
        usable_masks=usable_masks,
    )


def _sampling_masks_for(captured, frames):
    """Group the recorded sampling masks by the frame whose image was passed."""
    grouped = {}
    for image, mask in captured["sampling"]:
        for frame_id, frame in frames.items():
            if image is frame:
                grouped.setdefault(frame_id, []).append(mask)
                break
        else:  # pragma: no cover - a spy saw an image no frame owns
            raise AssertionError("sampling received an unknown image")
    return grouped


def test_explicit_masks_are_one_authority_for_every_consumer(monkeypatch):
    frames = {"a.png": _frame(SHAPE_A), "b.png": _frame(SHAPE_A)}
    mask_a = np.ones(SHAPE_A, dtype=bool)
    mask_a[10:40, 20:50] = False
    captured = _install_spies(monkeypatch)

    _reconstruct(frames, usable_masks={"a.png": mask_a})

    sampling = _sampling_masks_for(captured, frames)
    assert len(sampling["a.png"]) == 2, "both sampling passes must run"
    # Sampling gets the inverse of the authority, and the same object both times.
    for mask in sampling["a.png"]:
        assert mask.dtype == np.bool_
        assert not np.shares_memory(mask, mask_a)
        assert np.array_equal(mask, ~mask_a)
    assert len({id(mask) for mask in sampling["a.png"]}) == 1

    # Residual and fitting receive the one authority, by identity.
    residual_masks, fitting_masks = captured["residual"], captured["fitting"]
    assert residual_masks is not None and fitting_masks is not None
    assert residual_masks is fitting_masks, "one authority object, not two copies"
    assert set(residual_masks) == set(frames)
    assert residual_masks["a.png"] is mask_a
    assert np.array_equal(fitting_masks["a.png"], mask_a)


def test_a_frame_missing_from_the_mapping_is_unusable_everywhere(monkeypatch):
    frames = {"a.png": _frame(SHAPE_A), "b.png": _frame(SHAPE_A)}
    mask_a = np.ones(SHAPE_A, dtype=bool)
    captured = _install_spies(monkeypatch)

    _reconstruct(frames, usable_masks={"a.png": mask_a})

    authority = captured["residual"]
    assert authority is captured["fitting"]
    missing = authority["b.png"]
    assert missing.shape == SHAPE_A and missing.dtype == np.bool_
    assert not missing.any(), "a frame with no supplied mask has no usable pixel"
    # Sampling sees the complement: every pixel of that frame is excluded.
    sampling = _sampling_masks_for(captured, frames)
    assert len(sampling["b.png"]) == 2
    for mask in sampling["b.png"]:
        assert mask.dtype == np.bool_ and mask.all()


def test_an_empty_mapping_is_not_the_same_as_none(monkeypatch):
    """``{}`` is a supplied authority with every frame missing, not "no masks".

    An explicitly supplied empty map is authoritative under this contract. The
    benchmark caller now passes ``None`` when no frozen mask paths were given.
    """
    frames = {"a.png": _frame(SHAPE_A), "b.png": _frame(SHAPE_B)}
    captured = _install_spies(monkeypatch)

    _reconstruct(frames, usable_masks={})

    authority = captured["residual"]
    assert set(authority) == set(frames)
    for frame_id, shape in (("a.png", SHAPE_A), ("b.png", SHAPE_B)):
        assert authority[frame_id].shape == shape
        assert not authority[frame_id].any()
    sampling = _sampling_masks_for(captured, frames)
    for frame_id in frames:
        assert len(sampling[frame_id]) == 2
        assert all(mask.all() for mask in sampling[frame_id])

    # ``None`` is the derivable domain, so the two are not interchangeable: the
    # same frames are mostly usable once the mapping is dropped.
    derived = _install_spies(monkeypatch)
    _reconstruct(frames, usable_masks=None)
    assert derived["residual"] is not None
    assert derived["residual"]["a.png"].any()
    assert derived["residual"]["a.png"].shape == SHAPE_A


def test_a_wrong_shaped_mask_is_rejected_before_any_consumer(monkeypatch):
    frames = {"a.png": _frame(SHAPE_A)}
    wrong = np.ones((SHAPE_A[0] + 1, SHAPE_A[1]), dtype=bool)
    captured = _install_spies(monkeypatch)

    with pytest.raises(ValueError):
        _reconstruct(frames, usable_masks={"a.png": wrong})
    assert captured["sampling"] == []
    assert captured["residual"] is None and captured["fitting"] is None

    # Rejected even when the fitting consumer is switched off, so the check is
    # the call's authority and not a downstream shape test.
    with pytest.raises(ValueError):
        _reconstruct(frames, usable_masks={"a.png": wrong}, fit_partial_shapes=False)
    assert captured["sampling"] == []


def test_none_derives_the_gui_authority_for_every_consumer(monkeypatch):
    frames = {"a.png": _frame(SHAPE_A), "b.png": _frame(SHAPE_B)}
    captured = _install_spies(monkeypatch)

    _reconstruct(frames, usable_masks=None)

    # The derived map is passed on, not None, so no consumer can fall back to an
    # unmasked reading of the same frames.
    authority = captured["residual"]
    assert authority is not None and authority is captured["fitting"]
    assert set(authority) == set(frames)

    sampling = _sampling_masks_for(captured, frames)
    for frame_id, shape in (("a.png", SHAPE_A), ("b.png", SHAPE_B)):
        gui = create_gui_mask(*shape)
        derived = authority[frame_id]
        assert derived.shape == shape and derived.dtype == np.bool_
        assert np.array_equal(derived, ~gui)
        assert derived.any(), "a synthetic frame is mostly usable"
        assert len(sampling[frame_id]) == 2
        for mask in sampling[frame_id]:
            assert np.array_equal(mask, gui), "sampling keeps the historical HUD domain"
        assert len({id(mask) for mask in sampling[frame_id]}) == 1


def test_without_partial_fitting_only_sampling_reads_the_authority(monkeypatch):
    frames = {"a.png": _frame(SHAPE_A)}
    mask_a = np.ones(SHAPE_A, dtype=bool)
    mask_a[:, :10] = False
    captured = _install_spies(monkeypatch)

    _reconstruct(frames, usable_masks={"a.png": mask_a}, fit_partial_shapes=False)

    assert captured["residual"] is None and captured["fitting"] is None
    sampling = _sampling_masks_for(captured, frames)
    assert len(sampling["a.png"]) == 2
    for mask in sampling["a.png"]:
        assert np.array_equal(mask, ~mask_a)


# ---------------------------------------------------------------------------
# Actual-output controls: the production path, no spies.  A bounded synthetic
# frame whose fallback centre patches are the only evidence for its blocks.


BOUNDED_SHAPE = (128, 128)


def _bounded_palette():
    return (
        PaletteColor(BlockState("minecraft:black_concrete"), (10, 10, 10)),
        PaletteColor(BlockState("minecraft:stone"), (125, 125, 125)),
        PaletteColor(BlockState("minecraft:white_concrete"), (245, 245, 245)),
    )


def _bounded_scene():
    return SceneTransform(world_origin=(0, 0, 0), size=(8, 8, 8))


def _bounded_reconstruct(image, *, usable_masks, fit_partial_shapes=False, diagnostics=None):
    return raycast.reconstruct_scene_surfaces(
        {"a.png": image}, {"a.png": _camera()}, _bounded_scene(),
        palette=_bounded_palette(),
        propose_colour_gap=0.0,
        adjacency_strength=0.0,
        simulate_lighting=False,
        correct_light_cast=False,
        reconcile_exposure=False,
        self_calibrate=False,
        surface_only=False,
        fit_partial_shapes=fit_partial_shapes,
        extra_proposals={(0, 0, 0)},
        usable_masks=usable_masks,
        diagnostics=diagnostics,
    )


def _volume_signature(volume):
    return sorted(
        (coordinate, observation.state.name) for coordinate, observation in volume.items()
    )


def test_masked_fallback_pixels_cannot_change_the_output():
    """The excluded half is not read: fallback RGB invariance, measured."""
    mask = np.ones(BOUNDED_SHAPE, dtype=bool)
    mask[:, :64] = False
    usable = {"a.png": mask}
    dark = np.full((*BOUNDED_SHAPE, 3), 125, dtype=np.uint8)
    bright = dark.copy()
    dark[:, :64] = 5
    bright[:, :64] = 250

    dark_diagnostics: dict[str, object] = {}
    bright_diagnostics: dict[str, object] = {}
    dark_volume = _bounded_reconstruct(dark, usable_masks=usable, diagnostics=dark_diagnostics)
    bright_volume = _bounded_reconstruct(
        bright, usable_masks=usable, diagnostics=bright_diagnostics)

    # The control has to exercise the fallback, or equality would be vacuous.
    assert dark_diagnostics["fallback_samples"] > 0
    assert dark_diagnostics["fallback_samples"] == bright_diagnostics["fallback_samples"]
    assert _volume_signature(dark_volume) == _volume_signature(bright_volume)


@pytest.mark.parametrize("authority", ["missing", "all-false"])
def test_no_usable_pixels_yields_no_image_derived_evidence(authority):
    """All-false and missing authority both erase the centre-patch fallback."""
    rng = np.random.default_rng(7)
    image = rng.integers(1, 255, (*BOUNDED_SHAPE, 3), dtype=np.uint8)
    usable_masks = (
        {} if authority == "missing"
        else {"a.png": np.zeros(BOUNDED_SHAPE, dtype=bool)}
    )
    diagnostics: dict[str, object] = {}

    volume = _bounded_reconstruct(
        image, usable_masks=usable_masks, fit_partial_shapes=True, diagnostics=diagnostics)

    assert volume.blocks == {}
    assert diagnostics["resolved_face_samples"] == 0
    assert diagnostics["fallback_samples"] == 0
    assert diagnostics["blocks_from_fallback_only"] == 0
