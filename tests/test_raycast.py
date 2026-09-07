import numpy as np

from mc_photogrammetry.domain.models import BlockState
from mc_photogrammetry.reconstruction.camera import CameraModel
from mc_photogrammetry.reconstruction.raycast import reconstruct_scene_surfaces
from mc_photogrammetry.reconstruction.scene import SceneTransform
from mc_photogrammetry.reconstruction.facade import PaletteColor


def test_scene_raycast_recovers_visible_blocks_inside_bounds():
    image = np.full((128, 128, 3), 125, dtype=np.uint8)
    camera = CameraModel((4.0, 4.0, -10.0), 0.0, 0.0, eye_height=0.0)
    scene = SceneTransform((0, 0, 0), (8, 8, 8))
    palette = (PaletteColor(BlockState("minecraft:stone"), (125, 125, 125)),)

    volume = reconstruct_scene_surfaces({"frame.png": image}, {"frame.png": camera}, scene, palette=palette)

    assert volume.blocks
    assert all(0 <= coordinate[index] < scene.size[index] for coordinate in volume.blocks for index in range(3))

def test_a_badly_evidenced_cell_can_adopt_its_neighbours_label():
    """Adjacency alone can only re-rank a cell's own candidates.

    A cell whose evidence never proposed the right answer cannot be rescued by any
    amount of coupling, which is why `fallback_risk` measured inert from 0 to 40.
    `adopt_boost` admits the neighbours' majority as a candidate in its own right.
    Worth about 0.2pp when measured, so it ships off; this pins the mechanism.
    """
    import numpy as np

    from mc_photogrammetry.reconstruction.raycast import _smooth_adjacent_states

    right = BlockState("minecraft:stone")
    wrong = BlockState("minecraft:dirt")
    merged = {}
    for x in range(3):
        for z in range(3):
            centre = (x, 0, z) == (1, 0, 1)
            # The middle cell never considered `right` at all.
            scores = {wrong: 1.0} if centre else {right: 1.0, wrong: 0.1}
            merged[(x, 0, z)] = (wrong if centre else right, 0.0, set(),
                                 [wrong if centre else right], scores)

    without = {k: dict(v[4]) for k, v in merged.items()}
    _smooth_adjacent_states(merged, 1.0, 2, coupling_boost={(1, 0, 1): 5.0},
                            adopt_boost=0.0)
    assert max(merged[(1, 0, 1)][4], key=merged[(1, 0, 1)][4].get) == wrong

    merged = {k: (v[0], v[1], v[2], v[3], dict(without[k])) for k, v in merged.items()}
    _smooth_adjacent_states(merged, 1.0, 2, coupling_boost={(1, 0, 1): 5.0},
                            adopt_boost=2.0)
    assert max(merged[(1, 0, 1)][4], key=merged[(1, 0, 1)][4].get) == right
