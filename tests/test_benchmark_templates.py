import json

import numpy as np

from scripts.benchmark_templates import _load_usable_masks


def test_frozen_mask_spec_supports_compact_polygon_overlays(tmp_path):
    specification = tmp_path / "mask.json"
    specification.write_text(json.dumps({
        "image_size": [10, 8],
        "excluded_rectangles": [[0, 0, 2, 1]],
        "excluded_polygons": [[[4, 2], [7, 2], [4, 5]]],
    }), encoding="utf-8")

    mask = _load_usable_masks({"frame.png": specification}, {
        "frame.png": np.zeros((8, 10, 3), dtype=np.uint8),
    })["frame.png"]

    assert not mask[0, 0]
    assert not mask[3, 5]
    assert mask[6, 8]
