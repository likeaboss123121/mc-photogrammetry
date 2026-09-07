import numpy as np
import pytest

from mc_photogrammetry.reconstruction.raycast import reconstruct_scene_surfaces


def test_surface_contrast_rejects_flat_sample():
    with pytest.raises(ValueError, match="contrast"):
        reconstruct_scene_surfaces({}, {}, None, min_surface_contrast=-1)  # type: ignore[arg-type]