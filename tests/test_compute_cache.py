import numpy as np
import pytest

from mc_photogrammetry.compute import BoundedArtifactCache, FramePyramidCache, build_frame_pyramid


def test_frame_pyramid_is_half_resolution_and_read_only():
    pyramid = build_frame_pyramid(np.zeros((256, 192, 3), dtype=np.uint8), max_levels=4)

    assert [level.shape[:2] for level in pyramid.levels] == [
        (256, 192), (128, 96), (64, 48),
    ]
    assert pyramid.source_shape == (256, 192, 3)
    assert pyramid.nbytes == sum(level.nbytes for level in pyramid.levels)
    assert all(level.flags.writeable is False for level in pyramid.levels)


def test_frame_pyramid_cache_evicts_oldest_by_bytes():
    first = build_frame_pyramid(np.zeros((32, 32, 3), dtype=np.uint8), max_levels=1)
    second = build_frame_pyramid(np.ones((32, 32, 3), dtype=np.uint8), max_levels=1)
    cache = FramePyramidCache(max_bytes=first.nbytes)

    assert cache.put("first", first)
    assert cache.put("second", second)
    assert cache.get("first") is None
    assert cache.get("second") is not None
    assert cache.bytes_used == second.nbytes


def test_frame_pyramid_cache_updates_lru_order_and_rejects_invalid_limits():
    with pytest.raises(ValueError):
        FramePyramidCache(max_bytes=0)
    with pytest.raises(ValueError):
        build_frame_pyramid(np.zeros((4, 4), dtype=np.uint8), max_levels=0)

    pyramid = build_frame_pyramid(np.zeros((32, 32), dtype=np.uint8), max_levels=1)
    cache = FramePyramidCache(max_bytes=pyramid.nbytes * 2)
    cache.put("a", pyramid)
    cache.put("b", pyramid)
    cache.get("a")
    cache.put("c", pyramid)

    assert cache.get("a") is not None
    assert cache.get("b") is None
    assert len(cache) == 2


def test_bounded_artifact_cache_tracks_numpy_bytes_and_lru_order():
    cache = BoundedArtifactCache[np.ndarray](max_bytes=20)
    first = np.zeros(2, dtype=np.float32)
    second = np.ones(2, dtype=np.float32)
    third = np.ones(3, dtype=np.float32)

    assert cache.put("first", first)
    assert cache.put("second", second)
    assert cache.get("first") is first
    assert cache.put("third", third)
    assert cache.get("second") is None
    assert cache.get("first") is first
    assert cache.get("third") is third
    assert cache.bytes_used == 20


def test_bounded_artifact_cache_supports_explicit_size_and_rejects_unknown_values():
    cache = BoundedArtifactCache[str](max_bytes=4)
    assert cache.put("small", "artifact", size=4)
    assert cache.get("small") == "artifact"
    assert cache.put("too-large", "artifact", size=5) is False
    with pytest.raises(ValueError, match="nbytes"):
        cache.put("unknown", "artifact")
