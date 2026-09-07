from pathlib import Path
import zipfile

from PIL import Image

from mc_photogrammetry.blocks.textures import VanillaJarTextureProvider
from mc_photogrammetry.domain.models import BlockState
from mc_photogrammetry.reconstruction.raycast import _state_variant_prefix


def test_state_variant_prefix_keeps_material_roots_intact():
    assert _state_variant_prefix(BlockState("minecraft:dark_oak_planks")) == "dark_oak"
    assert _state_variant_prefix(BlockState("minecraft:birch_log")) == "birch"
    assert _state_variant_prefix(BlockState("minecraft:stone_brick_slab")) == "stone_brick"


def test_block_state_defaults_are_normalized_for_equivalence():
    assert BlockState("minecraft:oak_stairs") == BlockState("minecraft:oak_stairs", (("facing", "north"), ("half", "bottom"), ("shape", "straight"), ("waterlogged", "false")))
    assert BlockState("minecraft:birch_log") == BlockState("minecraft:birch_log", (("axis", "y"),))
    assert BlockState("minecraft:oak_slab") == BlockState("minecraft:oak_slab", (("type", "bottom"), ("waterlogged", "false")))


def test_texture_provider_extracts_local_block_texture(tmp_path):
    jar = tmp_path / "client.jar"
    texture = tmp_path / "stone.png"
    Image.new("RGBA", (2, 2), (125, 125, 125, 255)).save(texture)
    with zipfile.ZipFile(jar, "w") as archive:
        archive.write(texture, "assets/minecraft/textures/block/stone.png")
        archive.writestr(
            "assets/minecraft/blockstates/stone.json",
            '{"variants":{"":{"model":"minecraft:block/stone"}}}',
        )
        archive.writestr(
            "assets/minecraft/models/block/stone.json",
            '{"parent":"minecraft:block/cube_all","textures":{"all":"minecraft:block/stone"}}',
        )

    provider = VanillaJarTextureProvider()
    count = provider.load_jar(jar)

    assert count == 1
    assert provider.resolve("minecraft:stone").rgb == (125, 125, 125)
    assert provider.resolve("minecraft:stone").face_rgbs == ((125, 125, 125),)
    assert provider.resolve("minecraft:stone").family == "cube"


def test_texture_provider_preserves_variant_properties(tmp_path):
    jar = tmp_path / "client.jar"
    texture = tmp_path / "oak.png"
    Image.new("RGBA", (2, 2), (162, 130, 79, 255)).save(texture)
    with zipfile.ZipFile(jar, "w") as archive:
        archive.write(texture, "assets/minecraft/textures/block/oak_planks.png")
        archive.writestr(
            "assets/minecraft/blockstates/oak_stairs.json",
            '{"variants":{"facing=east,half=top":{"model":"minecraft:block/oak_stairs"}}}',
        )
        archive.writestr(
            "assets/minecraft/models/block/oak_stairs.json",
            '{"parent":"minecraft:block/cube_bottom_top","textures":{"bottom":"minecraft:block/oak_planks","top":"minecraft:block/oak_planks","side":"minecraft:block/oak_planks"}}',
        )

    provider = VanillaJarTextureProvider()
    provider.load_jar(jar)

    assert provider.resolve("minecraft:oak_stairs") is not None
    assert provider.resolve("minecraft:oak_stairs").state.properties == (
        ("facing", "east"),
        ("half", "top"),
        ("shape", "straight"),
        ("waterlogged", "false"),
    )
