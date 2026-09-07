import numpy as np
import pytest

from mc_photogrammetry.vision.gui_mask import (
    GuiMaskConfig,
    apply_manual_mask,
    create_gui_mask,
)


def test_gui_mask_scales_with_resolution():
    mask = create_gui_mask(100, 200)

    assert mask.shape == (100, 200)
    assert mask[99, 0]
    assert mask[50, 100]
    assert not mask[20, 199]


def test_manual_mask_does_not_mutate_original():
    original = np.zeros((10, 10), dtype=bool)
    updated = apply_manual_mask(original, x0=2, y0=3, x1=6, y1=7, value=True)

    assert not original.any()
    assert updated[3:7, 2:6].all()


def test_invalid_mask_dimensions_are_rejected():
    with pytest.raises(ValueError, match="dimensions"):
        create_gui_mask(0, 10, GuiMaskConfig())
