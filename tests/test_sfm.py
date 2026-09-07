import numpy as np
import pytest

from mc_photogrammetry.reconstruction.sfm import SfmSettings, reconstruct_from_calibrated_views


def test_sfm_requires_multiple_views():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="two calibrated frames"):
        reconstruct_from_calibrated_views(
            {"one.png": image}, {}, settings=SfmSettings()
        )