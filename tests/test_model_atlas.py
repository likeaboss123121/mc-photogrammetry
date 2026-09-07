import json
import zipfile

import numpy as np
import pytest
from PIL import Image

from mc_photogrammetry.blocks.model_atlas import (
    FACE_BASIS,
    FACE_SLOTS,
    ModelAtlas,
    blockstate_variants,
    variant_rotation,
)


def _column_jar(path, tmp_path):
    """A jar holding one pillar block with distinct end and side textures."""
    end = tmp_path / "end.png"
    side = tmp_path / "side.png"
    Image.new("RGBA", (16, 16), (200, 10, 10, 255)).save(end)
    Image.new("RGBA", (16, 16), (10, 10, 200, 255)).save(side)
    with zipfile.ZipFile(path, "w") as archive:
        archive.write(end, "assets/minecraft/textures/block/end.png")
        archive.write(side, "assets/minecraft/textures/block/side.png")
        archive.writestr(
            "assets/minecraft/blockstates/pillar.json",
            json.dumps(
                {
                    "variants": {
                        "axis=y": {"model": "minecraft:block/pillar"},
                        "axis=z": {"model": "minecraft:block/pillar", "x": 90},
                        "axis=x": {"model": "minecraft:block/pillar", "x": 90, "y": 90},
                    }
                }
            ),
        )
        archive.writestr(
            "assets/minecraft/models/block/pillar.json",
            json.dumps(
                {
                    "textures": {"end": "minecraft:block/end", "side": "minecraft:block/side"},
                    "elements": [
                        {
                            "from": [0, 0, 0],
                            "to": [16, 16, 16],
                            "faces": {
                                "down": {"texture": "#end"},
                                "up": {"texture": "#end"},
                                "north": {"texture": "#side"},
                                "south": {"texture": "#side"},
                                "west": {"texture": "#side"},
                                "east": {"texture": "#side"},
                            },
                        }
                    ],
                }
            ),
        )


def _end_slots(baked):
    return {slot for slot, face in baked.items() if face.rgb[0] > face.rgb[2]}


def test_variant_rotation_moves_end_faces_onto_the_right_axis(tmp_path):
    jar = tmp_path / "client.jar"
    _column_jar(jar, tmp_path)
    with zipfile.ZipFile(jar) as archive:
        atlas = ModelAtlas(archive)
        blockstate = json.loads(archive.read("assets/minecraft/blockstates/pillar.json"))
        baked = {
            dict(properties)["axis"]: atlas.bake(model, x, y)
            for properties, model, x, y in blockstate_variants(blockstate)
        }

    assert _end_slots(baked["y"]) == {"up", "down"}
    assert _end_slots(baked["z"]) == {"north", "south"}
    assert _end_slots(baked["x"]) == {"east", "west"}


def test_baked_faces_cover_every_slot_with_matching_corners(tmp_path):
    jar = tmp_path / "client.jar"
    _column_jar(jar, tmp_path)
    with zipfile.ZipFile(jar) as archive:
        baked = ModelAtlas(archive).bake("minecraft:block/pillar")

    assert set(baked) == set(FACE_SLOTS)
    for slot, face in baked.items():
        assert face.texture.shape == (16, 16, 3)
        assert face.corners.shape == (4, 3)
        # The four corners lie on the face's own plane, one block-half from centre.
        normal = np.asarray(FACE_BASIS[slot][0], dtype=float)
        assert np.allclose(face.corners @ normal, 0.5)


def test_rotation_matches_vanilla_sign_convention():
    # Vanilla rotates by -x about +X, so a 90 degree x turn sends up to north.
    rotated = variant_rotation(90.0, 0.0) @ np.asarray([0.0, 1.0, 0.0])
    assert np.allclose(rotated, [0.0, 0.0, -1.0], atol=1e-9)


def test_bake_rejects_models_without_full_cube_geometry(tmp_path):
    jar = tmp_path / "client.jar"
    texture = tmp_path / "flat.png"
    Image.new("RGBA", (16, 16), (1, 2, 3, 255)).save(texture)
    with zipfile.ZipFile(jar, "w") as archive:
        archive.write(texture, "assets/minecraft/textures/block/flat.png")
        archive.writestr(
            "assets/minecraft/models/block/carpet.json",
            json.dumps(
                {
                    "textures": {"all": "minecraft:block/flat"},
                    "elements": [{"from": [0, 0, 0], "to": [16, 1, 16], "faces": {"up": {"texture": "#all"}}}],
                }
            ),
        )
    with zipfile.ZipFile(jar) as archive:
        assert ModelAtlas(archive).bake("minecraft:block/carpet") is None


@pytest.mark.parametrize("missing", ["assets/minecraft/models/block/absent.json"])
def test_bake_returns_none_for_unknown_models(tmp_path, missing):
    jar = tmp_path / "client.jar"
    with zipfile.ZipFile(jar, "w") as archive:
        archive.writestr("assets/minecraft/models/block/present.json", "{}")
    with zipfile.ZipFile(jar) as archive:
        assert ModelAtlas(archive).bake("minecraft:block/absent") is None
