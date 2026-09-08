import json
from pathlib import Path
import zipfile

from PIL import Image

from mc_photogrammetry.blocks.textures import CACHE_SCHEMA_VERSION, VanillaJarTextureProvider


def test_texture_provider_reuses_jar_cache(tmp_path):
    jar = tmp_path / "client.jar"
    texture = tmp_path / "stone.png"
    Image.new("RGBA", (2, 2), (125, 125, 125, 255)).save(texture)
    with zipfile.ZipFile(jar, "w") as archive:
        archive.write(texture, "assets/minecraft/textures/block/stone.png")
        archive.writestr("assets/minecraft/blockstates/stone.json", '{"variants":{"":{"model":"minecraft:block/stone"}}}')
        archive.writestr("assets/minecraft/models/block/stone.json", '{"textures":{"all":"minecraft:block/stone"}}')

    first = VanillaJarTextureProvider()
    first.load_jar(jar)
    assert first.source == "extracted"
    cache = Path(str(jar).replace(".jar", ".mc-textures.json"))
    assert cache.exists()
    # Compare against the constant: pinning a literal here means every schema
    # bump breaks this test for no reason, and the thing worth asserting is that
    # the cache records whatever schema the code currently writes.
    assert json.loads(cache.read_text(encoding="utf-8"))["cache_schema"] == CACHE_SCHEMA_VERSION

    second = VanillaJarTextureProvider()
    assert second.load_jar(jar) == first.load_jar(jar)
    assert second.source == "cached"


def test_empty_cache_triggers_reextraction(tmp_path):
    jar = tmp_path / "client.jar"
    texture = tmp_path / "stone.png"
    Image.new("RGBA", (2, 2), (125, 125, 125, 255)).save(texture)
    with zipfile.ZipFile(jar, "w") as archive:
        archive.write(texture, "assets/minecraft/textures/block/stone.png")
        archive.writestr("assets/minecraft/blockstates/stone.json", '{"variants":{"":{"model":"minecraft:block/stone"}}}')
        archive.writestr("assets/minecraft/models/block/stone.json", '{"textures":{"all":"minecraft:block/stone"}}')

    cache = jar.with_suffix(".mc-textures.json")
    cache.write_text(
        json.dumps(
            {
                "cache_schema": 4,
                "jar_size": jar.stat().st_size,
                "jar_mtime": jar.stat().st_mtime_ns,
                "samples": [],
            }
        ),
        encoding="utf-8",
    )

    provider = VanillaJarTextureProvider()
    loaded = provider.load_jar(jar)

    assert loaded > 0
    assert provider.samples
    assert provider.source == "extracted"
    assert provider.samples[0].state.name == "minecraft:stone"


def test_face_model_resolves_only_visible_face_textures(tmp_path):
    jar = tmp_path / "client.jar"
    top = tmp_path / "top.png"
    side = tmp_path / "side.png"
    bottom = tmp_path / "bottom.png"
    unused = tmp_path / "unused.png"
    Image.new("RGBA", (2, 2), (10, 20, 30, 255)).save(top)
    Image.new("RGBA", (2, 2), (40, 50, 60, 255)).save(side)
    Image.new("RGBA", (2, 2), (70, 80, 90, 255)).save(bottom)
    Image.new("RGBA", (2, 2), (100, 110, 120, 255)).save(unused)

    with zipfile.ZipFile(jar, "w") as archive:
        archive.write(top, "assets/minecraft/textures/block/top.png")
        archive.write(side, "assets/minecraft/textures/block/side.png")
        archive.write(bottom, "assets/minecraft/textures/block/bottom.png")
        archive.write(unused, "assets/minecraft/textures/block/unused.png")
        archive.writestr(
            "assets/minecraft/blockstates/faced_block.json",
            '{"variants":{"":{"model":"minecraft:block/faced_block"}}}',
        )
        archive.writestr(
            "assets/minecraft/models/block/faced_block.json",
            '{"textures":{"particle":"minecraft:block/unused","top":"minecraft:block/top","bottom":"minecraft:block/bottom","north":"minecraft:block/side","south":"minecraft:block/side","east":"minecraft:block/side","west":"minecraft:block/side"},"elements":[{"from":[0,0,0],"to":[16,16,16],"faces":{"up":{"texture":"#top"},"down":{"texture":"#bottom"},"north":{"texture":"#north"},"south":{"texture":"#south"},"east":{"texture":"#east"},"west":{"texture":"#west"}}}]}',
        )

    provider = VanillaJarTextureProvider()
    provider.load_jar(jar)

    sample = provider.resolve("minecraft:faced_block")
    assert sample is not None
    assert sample.face_paths
    assert "assets/minecraft/textures/block/unused.png" not in sample.face_paths
    assert sample.face_paths[:3] == (
        "assets/minecraft/textures/block/top.png",
        "assets/minecraft/textures/block/bottom.png",
        "assets/minecraft/textures/block/side.png",
    )
    assert dict(sample.face_map) == {
        "up": (10, 20, 30),
        "down": (70, 80, 90),
        "north": (40, 50, 60),
        "south": (40, 50, 60),
        "east": (40, 50, 60),
        "west": (40, 50, 60),
    }
    assert len(sample.model_faces) == 6
    assert {face[0] for face in sample.model_faces} == {"up", "down", "north", "south", "east", "west"}
    assert dict(sample.face_thumbnails)
