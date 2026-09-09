import numpy as np
import pytest

from mc_photogrammetry.compute import (
    RenderBufferLayout,
    pack_render_result,
    unpack_render_buffers,
)
from mc_photogrammetry.pipeline.contracts import RenderResult


def test_pack_render_result_canonicalises_dtypes_and_derives_validity():
    ownership = np.array([[0, -1, 4], [2, -1, 9]], dtype=np.int64)
    result = RenderResult(
        colour=np.arange(18, dtype=np.float64).reshape(2, 3, 3),
        depth=np.array([[1, np.inf, 3], [4, 5, 6]], dtype=np.float64),
        ownership=ownership,
    )

    packed = pack_render_result(result)

    assert packed.layout == RenderBufferLayout(2, 3, False)
    assert packed.colour.dtype == np.float32
    assert packed.depth.dtype == np.float32
    assert packed.ownership.dtype == np.int32
    assert packed.valid_pixels.dtype == np.uint8
    np.testing.assert_array_equal(packed.valid_pixels, [[1, 0, 1], [1, 0, 1]])
    assert not packed.colour.flags.writeable
    assert result.ownership.dtype == np.int64


def test_pack_render_result_keeps_uv_and_explicit_validity():
    result = RenderResult(
        colour=np.zeros((2, 2, 3), dtype=np.float32),
        depth=np.ones((2, 2), dtype=np.float32),
        ownership=np.zeros((2, 2), dtype=np.int32),
        uv=np.zeros((2, 2, 2), dtype=np.float64),
        valid_pixels=np.array([[True, False], [False, True]]),
    )

    packed = pack_render_result(result)

    assert packed.layout.has_uv
    assert packed.uv is not None and packed.uv.dtype == np.float32
    np.testing.assert_array_equal(packed.valid_pixels, [[1, 0], [0, 1]])
    roundtrip = unpack_render_buffers(packed)
    np.testing.assert_array_equal(roundtrip.colour, result.colour)
    np.testing.assert_array_equal(roundtrip.uv, result.uv)
    np.testing.assert_array_equal(roundtrip.valid_pixels, result.valid_pixels)


def test_pack_render_result_owns_copies_and_does_not_change_source_flags():
    colour = np.zeros((2, 2, 3), dtype=np.float32)
    colour_alias = colour.view()
    result = RenderResult(
        colour=colour,
        depth=np.ones((2, 2), dtype=np.float32),
        ownership=np.zeros((2, 2), dtype=np.int32),
    )

    packed = pack_render_result(result)

    assert colour.flags.writeable
    colour_alias[0, 0, 0] = 99.0
    assert packed.colour[0, 0, 0] == 0.0


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("colour", np.zeros((2, 2), dtype=np.float32), "colour"),
        ("depth", np.zeros((3, 2), dtype=np.float32), "matching"),
        ("ownership", np.zeros((2, 2), dtype=np.float32), "integer"),
        ("uv", np.zeros((2, 2), dtype=np.float32), "uv"),
    ],
)
def test_pack_render_result_rejects_invalid_shapes_or_types(field, value, message):
    kwargs = {
        "colour": np.zeros((2, 2, 3), dtype=np.float32),
        "depth": np.zeros((2, 2), dtype=np.float32),
        "ownership": np.zeros((2, 2), dtype=np.int32),
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=message):
        pack_render_result(RenderResult(**kwargs))


def test_layout_reports_unpadded_wire_sizes():
    layout = RenderBufferLayout(3, 5, True)

    assert layout.shape == (3, 5)
    assert layout.byte_sizes == {
        "colour": 180, "depth": 60, "ownership": 60, "valid_pixels": 15, "uv": 120,
    }
