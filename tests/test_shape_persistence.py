import json
import zipfile

from PIL import Image

from mc_photogrammetry.blocks.textures import CACHE_SCHEMA_VERSION, VanillaJarTextureProvider
from mc_photogrammetry.domain.models import BlockState

FACE_SLOTS = ("down", "up", "north", "south", "west", "east")


def _element(start, end):
    return {
        "from": start,
        "to": end,
        "faces": {slot: {"texture": "#all"} for slot in FACE_SLOTS},
    }


def _jar(path, tmp_path):
    """A jar holding a full cube, a bottom slab, a carpet and a straight stair."""
    texture = tmp_path / "all.png"
    Image.new("RGBA", (2, 2), (120, 90, 60, 255)).save(texture)
    models = {
        "full_block": [_element([0, 0, 0], [16, 16, 16])],
        "slab_block": [_element([0, 0, 0], [16, 8, 16])],
        "carpet_block": [_element([0, 0, 0], [16, 1, 16])],
        # Lower slab plus a half-depth upper step, the shape of a staircase.
        "stair_block": [_element([0, 0, 0], [16, 8, 16]), _element([0, 8, 8], [16, 16, 16])],
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.write(texture, "assets/minecraft/textures/block/all.png")
        for name, elements in models.items():
            archive.writestr(
                f"assets/minecraft/blockstates/{name}.json",
                json.dumps({"variants": {"": {"model": f"minecraft:block/{name}"}}}),
            )
            archive.writestr(
                f"assets/minecraft/models/block/{name}.json",
                json.dumps({"textures": {"all": "minecraft:block/all"}, "elements": elements}),
            )


def _provider(tmp_path):
    jar = tmp_path / "shapes.jar"
    _jar(jar, tmp_path)
    provider = VanillaJarTextureProvider()
    provider.load_jar(jar)
    return provider, jar


def test_shape_is_recorded_for_partial_models(tmp_path):
    provider, _jar_path = _provider(tmp_path)
    slab = provider.resolve("minecraft:slab_block")
    carpet = provider.resolve("minecraft:carpet_block")
    assert slab is not None and carpet is not None
    assert slab.shape_boxes and carpet.shape_boxes
    assert slab.shape_height == 0.5
    assert round(carpet.shape_height, 4) == 0.0625
    # A carpet is a sixteenth of a block tall and still hides the face beneath it.
    assert carpet.occludes("down")
    assert not carpet.occludes("up")


def test_a_full_cube_occludes_every_direction(tmp_path):
    provider, _jar_path = _provider(tmp_path)
    block = provider.resolve("minecraft:full_block")
    assert block is not None
    assert block.full_cube
    assert all(block.occludes(slot) for slot in FACE_SLOTS)


def test_a_stair_shape_does_not_occlude_upward(tmp_path):
    provider, _jar_path = _provider(tmp_path)
    stair = provider.resolve("minecraft:stair_block")
    assert stair is not None
    # Its bounding box fills the cell, so judging from bounds would wrongly
    # report that it hides the block above it.
    assert not stair.full_cube
    assert stair.occludes("down")
    assert not stair.occludes("up")


def test_shape_survives_the_cache_round_trip(tmp_path):
    provider, jar = _provider(tmp_path)
    assert provider.source == "extracted"
    reloaded = VanillaJarTextureProvider()
    reloaded.load_jar(jar)
    assert reloaded.source == "cached"
    for name in ("full_block", "slab_block", "carpet_block", "stair_block"):
        first = provider.resolve(f"minecraft:{name}")
        second = reloaded.resolve(f"minecraft:{name}")
        assert second is not None
        assert second.shape_boxes == first.shape_boxes
        assert second.covered_slots == first.covered_slots


def test_a_stale_schema_forces_re_extraction(tmp_path):
    provider, jar = _provider(tmp_path)
    caches = [
        path
        for path in jar.parent.rglob("*.json")
        if "cache_schema" in path.read_text(encoding="utf-8")
    ]
    assert caches, "expected a texture cache file to have been written"
    for path in caches:
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["cache_schema"] == CACHE_SCHEMA_VERSION
        data["cache_schema"] = CACHE_SCHEMA_VERSION - 1
        path.write_text(json.dumps(data), encoding="utf-8")
    again = VanillaJarTextureProvider()
    again.load_jar(jar)
    assert again.source == "extracted"


def test_partial_states_stay_out_of_the_full_cube_palette(tmp_path):
    provider, _jar_path = _provider(tmp_path)
    names = {item.state.name for item in provider.canonical_palette()}
    # Shape is recorded so partial blocks can occlude; that must not let them
    # become classifier candidates, which compare complete-cube appearances.
    assert "minecraft:full_block" in names
    assert "minecraft:slab_block" not in names
    assert "minecraft:carpet_block" not in names
    assert "minecraft:stair_block" not in names


def test_palette_for_states_keeps_exact_partial_renderer_metadata(tmp_path):
    provider, _jar_path = _provider(tmp_path)
    requested = (BlockState("minecraft:slab_block"),)

    selected = provider.palette_for_states(requested)

    assert len(selected) == 1
    slab = selected[0]
    assert slab.state == requested[0]
    assert slab.full_cube is False
    assert slab.shape_boxes
    assert slab.model_faces
    assert slab.covered_slots
    assert provider.palette_for_states((BlockState("minecraft:missing"),)) == ()
