import numpy as np
import pytest

from mc_photogrammetry.input.frames import (
    FrameCandidate,
    hamming_distance,
    perceptual_hash,
    quality_score,
    select_keyframes,
)


def test_identical_images_have_zero_hash_distance():
    image = np.full((16, 16, 3), 120, dtype=np.uint8)

    assert hamming_distance(perceptual_hash(image), perceptual_hash(image.copy())) == 0


def test_keyframe_selection_removes_duplicate_images_and_preserves_source_order():
    gradient = np.arange(16, dtype=np.uint8)
    first = np.broadcast_to(gradient, (16, 16))[..., None].repeat(3, axis=2).copy()
    second = np.flip(first, axis=1).copy()
    frames = [
        FrameCandidate("first", first),
        FrameCandidate("duplicate", first.copy()),
        FrameCandidate("second", second),
    ]

    selected = select_keyframes(frames, hash_distance=0)

    assert [frame.frame_id for frame in selected] == ["first", "second"]


def test_quality_score_rejects_invalid_shape():
    with pytest.raises(ValueError, match="RGB uint8"):
        quality_score(np.zeros((4, 4), dtype=np.uint8))
