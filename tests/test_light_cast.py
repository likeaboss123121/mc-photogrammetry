"""Undoing the light on a dim scene, colour as well as brightness.

Minecraft block light is warm, so a dim room is not merely darker than a bright
one - it is redder, by a per-channel amount whose normalised shape measured the
same on all three gamma-100 fixtures and flat on the fullbright one. That makes
the cast a constant and its magnitude the only unknown, which is what these two
functions recover: `_scene_spread` decides whether there is any light to undo, and
`_uncast` undoes it.

Both work from the pixels alone. That is the point: `_scene_brightness`, which
asks the palette how bright its chosen match should have been, reported Template 5
as fullbright at 1.006 when its true gain was 0.401, because that fixture's palette
holds a convincing dark block for every material in it.
"""
import numpy as np
import pytest

from mc_photogrammetry.reconstruction.raycast import (
    CORRECT_LIGHT_CAST,
    LIGHT_CAST,
    _chroma_key,
    _scene_spread,
    _uncast,
)


def _observation(rgb, slot="north", coordinate=(0, 0, 0), area=64.0, frame="a.png"):
    return (frame, coordinate, np.asarray(rgb, dtype=np.float32), (), slot, 1.0, None, area)


def _many(rgb, count, slot="north"):
    return [_observation(rgb, slot, (i, 0, 0)) for i in range(count)]


def test_a_colour_keeps_its_key_however_dark_it_is():
    bright = np.asarray([200.0, 120.0, 60.0])
    assert _chroma_key(bright) == _chroma_key(bright * 0.3)
    assert _chroma_key(bright) == _chroma_key(bright * 0.05)


def test_chroma_spread_light_correction_is_the_pipeline_default():
    """The self-labelled brightness gate regressed Template 6 to zero states."""
    assert CORRECT_LIGHT_CAST is True


def test_different_colours_get_different_keys():
    assert _chroma_key(np.asarray([200.0, 40.0, 40.0])) != _chroma_key(
        np.asarray([40.0, 40.0, 200.0])
    )


def test_a_black_face_does_not_divide_by_zero():
    assert _chroma_key(np.asarray([0.0, 0.0, 0.0])) == (0, 0, 0)


def test_an_evenly_lit_scene_has_no_spread():
    # Every face of the material reads the same, which is what fullbright means.
    assert _scene_spread(_many([160.0, 100.0, 50.0], 40)) == pytest.approx(1.0, abs=0.01)


def test_face_shading_alone_is_not_mistaken_for_light():
    # Vanilla shades a north face to 0.8 and an east face to 0.6 regardless of
    # lighting, so a scene lit perfectly evenly must still read as unlit.
    base = np.asarray([160.0, 100.0, 50.0])
    raw = (
        [_observation(base * 1.0, "up", (i, 0, 0)) for i in range(20)]
        + [_observation(base * 0.8, "north", (i, 1, 0)) for i in range(20)]
        + [_observation(base * 0.6, "east", (i, 2, 0)) for i in range(20)]
    )
    assert _scene_spread(raw) == pytest.approx(1.0, abs=0.05)


def test_one_material_at_many_brightnesses_reads_as_lit():
    base = np.asarray([160.0, 100.0, 50.0])
    raw = [
        _observation(base * factor, "north", (i, 0, 0))
        for i, factor in enumerate(np.linspace(0.2, 1.0, 40))
    ]
    assert _scene_spread(raw) > 2.0


def test_faces_with_no_area_are_ignored():
    # The centre-patch fallback carries no rectified face, so it says nothing
    # about how the scene is lit.
    raw = _many([160.0, 100.0, 50.0], 20) + [
        _observation([10.0, 6.0, 3.0], "north", (99, 0, 0), area=0.0) for _ in range(20)
    ]
    assert _scene_spread(raw) == pytest.approx(1.0, abs=0.01)


def test_a_dimmed_face_is_lifted_towards_its_brightest_twin():
    base = np.asarray([200.0, 160.0, 120.0])
    raw = _many(base, 20) + _many(base * 0.4, 20)
    lifted = {tuple(np.round(o[2] / np.mean(o[2]), 3)) for o in _uncast(raw)}
    # Both halves now describe the same material, so one direction survives.
    assert len(lifted) == 1


def test_the_brightest_face_of_a_colour_is_left_where_it_is():
    base = np.asarray([200.0, 160.0, 120.0])
    raw = _many(base, 20) + _many(base * 0.4, 20)
    corrected = _uncast(raw, cast=(1.0, 1.0, 1.0))
    assert np.allclose(corrected[0][2], base, rtol=0.02)


def test_the_colour_cast_is_undone_and_not_only_the_brightness():
    # A white block under warm light reads orange. Scaling it up keeps it orange;
    # only dividing by the cast makes it white again, and that is the difference
    # between 1.88% and 37.59% per-face accuracy on Template 4.
    white = np.asarray([200.0, 200.0, 200.0])
    dim = white * 0.4 * np.asarray(LIGHT_CAST)
    raw = _many(white, 20) + _many(dim, 20)
    corrected = _uncast(raw)[-1][2]
    scaled = dim / (np.mean(dim) / np.mean(white))
    assert abs(float(corrected[0] - corrected[2])) < abs(float(scaled[0] - scaled[2]))


def test_a_colour_seen_too_few_times_is_left_alone():
    # One face of a colour says nothing about how bright that material can be, so
    # inventing a ceiling from it would correct against noise. A *dim* face of a
    # well-seen colour is a different matter and is corrected: chroma survives
    # dimming, so it lands in that colour's group by design.
    common = np.asarray([200.0, 160.0, 120.0])
    rare = np.asarray([30.0, 40.0, 200.0])
    raw = _many(common, 20) + [_observation(rare, "north", (99, 0, 0))]
    # Isolated from the cast, which is scene-wide and applies to every face.
    assert np.allclose(_uncast(raw, cast=(1.0, 1.0, 1.0))[-1][2], rare, rtol=0.02)


def test_the_cast_applies_to_every_face_of_a_lit_scene():
    # The cast is a property of the light in the room, not of one face's distance
    # from a torch, so it is divided out wherever the scene is judged lit.
    base = np.asarray([200.0, 160.0, 120.0])
    corrected = _uncast(_many(base, 20))[0][2]
    assert not np.allclose(corrected, base, rtol=0.02)
    assert corrected[0] < base[0] and corrected[2] > base[2]


def test_uncast_preserves_the_shape_of_every_observation():
    raw = _many([200.0, 160.0, 120.0], 20)
    corrected = _uncast(raw)
    assert len(corrected) == len(raw)
    for before, after in zip(raw, corrected):
        assert before[0] == after[0] and before[1] == after[1]
        assert before[4] == after[4] and before[7] == after[7]
