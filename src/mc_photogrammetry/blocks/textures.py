from __future__ import annotations

import json
import base64
from io import BytesIO
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mc_photogrammetry.domain.models import BlockState
from mc_photogrammetry.reconstruction.facade import PaletteColor
from .model_atlas import (
    FACE_SLOTS,
    ModelAtlas,
    blockstate_multipart,
    blockstate_variants,
    compose,
)

CACHE_SCHEMA_VERSION = 15

COLOUR_RESOLUTION = 4

PATTERN_CELLS = 4
"""Grid a face texture is pooled to when describing its pattern."""

PATTERN_LEVELS = 2
"""Quantisation of the pooled pattern. Zero ignores pattern entirely."""

PATTERN_SPREAD_STEP = 12.0
"""Quantisation of a face's overall contrast, which separates a busy
texture from a flat one even when their layouts agree."""
"""Colour levels within which two block faces cannot be told apart in a screenshot.

Kept small: collapsing harder drops real blocks. At 8 the palette loses
``polished_blackstone_bricks``, which is 21% of Template 4.
"""


@dataclass(frozen=True, slots=True)
class TextureSample:
    state: BlockState
    rgb: tuple[int, int, int]
    texture_path: str
    face_paths: tuple[str, ...] = ()
    face_rgbs: tuple[tuple[int, int, int], ...] = ()
    family: str = "other"
    descriptors: tuple[tuple[float, ...], ...] = ()
    full_cube: bool = False
    face_map: tuple[tuple[str, tuple[int, int, int]], ...] = ()
    face_thumbnails: tuple[tuple[str, str], ...] = ()
    face_corners: tuple[tuple[str, tuple[tuple[float, float, float], ...]], ...] = ()
    model_faces: tuple[tuple, ...] = ()
    """Exact polygons: slot, corners, normal, cullface, and optional encoded alpha mask."""
    shape_boxes: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...] = ()
    """Each model element's own extent in block units, after the variant rotation.

    Kept separately from the full-cube face appearance above, because shape and
    appearance answer different questions: the classifier compares appearances of
    complete cubes, while occlusion needs to know that a carpet one sixteenth of a
    block tall still hides the face it rests on.
    """
    covered_slots: tuple[str, ...] = ()
    """World faces of the cell this model completely hides."""

    @property
    def shape_height(self) -> float:
        """How tall the model stands, or zero when its shape was not resolved."""
        if not self.shape_boxes:
            return 0.0
        return max(box[1][1] for box in self.shape_boxes) - min(box[0][1] for box in self.shape_boxes)

    def occludes(self, slot: str) -> bool:
        """Whether this block hides its neighbour in the given direction."""
        return slot in self.covered_slots


class VanillaJarTextureProvider:
    """Read local vanilla block textures without bundling Mojang assets."""

    def __init__(self) -> None:
        self.samples: list[TextureSample] = []
        self.jar_path: Path | None = None
        self.source: str | None = None

    def load_jar(self, path: str | Path) -> int:
        try:
            from PIL import Image
        except ImportError as error:
            raise RuntimeError("texture extraction requires Pillow") from error

        jar_path = Path(path)
        if not jar_path.is_file():
            raise FileNotFoundError(jar_path)
        cache_path = jar_path.with_suffix(".mc-textures.json")
        cached = _load_cached_samples(cache_path, jar_path)
        if cached:
            self.samples = cached
            self.jar_path = jar_path.resolve()
            self.source = "cached"
            return len(self.samples)
        samples: list[TextureSample] = []
        with zipfile.ZipFile(jar_path) as archive:
            texture_cache: dict[str, tuple[int, int, int]] = {}
            texture_descriptors: dict[str, tuple[float, ...]] = {}
            for path_name in archive.namelist():
                    if not path_name.startswith("assets/minecraft/textures/block/") or not path_name.endswith(".png"):
                        continue
                    try:
                        with Image.open(archive.open(path_name)) as image:
                            rgba = np.asarray(image.convert("RGBA"), dtype=np.float32)
                    except (OSError, ValueError):
                        continue
                    alpha = rgba[:, :, 3] / 255.0
                    weight = alpha.sum()
                    if weight > 0:
                        texture_cache[path_name] = tuple(
                            int(round(value))
                            for value in (rgba[:, :, :3] * alpha[:, :, None]).sum(axis=(0, 1)) / weight
                        )
                        texture_descriptors[path_name] = _texture_descriptor(rgba, alpha)
                        texture_cache[f"__pixels__{path_name}"] = _thumbnail_pixels(rgba)

            atlas = ModelAtlas(archive)
            for blockstate_path in archive.namelist():
                    prefix = "assets/minecraft/blockstates/"
                    if not blockstate_path.startswith(prefix) or not blockstate_path.endswith(".json"):
                        continue
                    block_name = Path(blockstate_path).stem
                    try:
                        blockstate = json.loads(archive.read(blockstate_path))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                    for properties, model_name, x_degrees, y_degrees in blockstate_variants(blockstate):
                        # Baking applies the variant's model rotation, so a horizontal
                        # log reports end grain on the world faces that really show it.
                        baked = atlas.bake(model_name, x_degrees, y_degrees)
                        full_cube = baked is not None
                        # Exact polygons carry their own cropped alpha masks, so
                        # cut-out blocks (leaves, panes, plants) do not become
                        # opaque in the attribution renderer.
                        shape = atlas.geometry(model_name, x_degrees, y_degrees)
                        shape_boxes = () if shape is None else shape.boxes
                        covered_slots = (
                            ()
                            if shape is None
                            else tuple(slot for slot in FACE_SLOTS if shape.covers(slot))
                        )
                        model_faces = () if shape is None else tuple(
                            (face.slot,
                             tuple(tuple(float(v) for v in corner) for corner in face.corners),
                             tuple(float(v) for v in face.normal), face.cullface,
                             _encode_alpha_mask(face.alpha))
                            for face in shape.faces
                        )
                        face_slots = _resolve_face_texture_slots(archive, model_name, set())
                        face_paths = _resolve_face_textures(archive, model_name, set()) or _resolve_model_textures(archive, model_name, set())
                        unique_paths = tuple(dict.fromkeys(face_paths))
                        colors = [texture_cache[path_name] for path_name in unique_paths if path_name in texture_cache]
                        if not colors:
                            continue
                        rgb = tuple(int(round(sum(color[index] for color in colors) / len(colors))) for index in range(3))
                        descriptors = tuple(texture_descriptors[path_name] for path_name in unique_paths if path_name in texture_descriptors)
                        if baked is not None:
                            face_map = tuple((slot, baked[slot].rgb) for slot in FACE_SLOTS)
                            face_thumbnails = tuple(
                                (slot, _encode_thumbnail(baked[slot].texture)) for slot in FACE_SLOTS
                            )
                            face_corners = tuple(
                                (slot, tuple(tuple(float(v) for v in corner) for corner in baked[slot].corners))
                                for slot in FACE_SLOTS
                            )
                            rgb = tuple(
                                int(round(sum(baked[slot].rgb[index] for slot in FACE_SLOTS) / len(FACE_SLOTS)))
                                for index in range(3)
                            )
                        else:
                            face_map = tuple(
                                (slot, texture_cache[path_name])
                                for slot, path_name in face_slots.items()
                                if path_name in texture_cache
                            )
                            face_thumbnails = tuple(
                                (slot, _encode_thumbnail(texture_cache[f"__pixels__{path_name}"]))
                                for slot, path_name in face_slots.items()
                                if f"__pixels__{path_name}" in texture_cache
                            )
                            face_corners = ()
                        samples.append(
                            TextureSample(
                                BlockState(
                                    f"minecraft:{block_name}",
                                    _complete_properties(block_name, properties),
                                ),
                                rgb,
                                unique_paths[0],
                                unique_paths,
                                tuple(colors),
                                _model_family(block_name, list(unique_paths)),
                                descriptors,
                                full_cube,
                                face_map,
                                face_thumbnails,
                                face_corners,
                                model_faces,
                                shape_boxes,
                                covered_slots,
                            )
                        )

                    # Fences, walls, panes and bars are described as a set of
                    # pieces with conditions rather than as variants, so the
                    # variant reader above sees nothing and they never reach the
                    # palette at all. Compose the pieces that apply and record the
                    # resulting shape.
                    for properties, parts in blockstate_multipart(blockstate):
                        shape = compose([
                            atlas.geometry(model, x_degrees, y_degrees)
                            for model, x_degrees, y_degrees in parts
                        ])
                        if shape is None or not shape.boxes:
                            continue
                        textured = [face for face in shape.faces if face.texture is not None]
                        if not textured:
                            continue
                        by_slot: dict[str, list] = {}
                        for face in textured:
                            by_slot.setdefault(face.slot, []).append(face.texture)
                        face_map = tuple(
                            (
                                slot,
                                tuple(
                                    int(round(value))
                                    for value in np.mean(
                                        [t.reshape(-1, 3).mean(axis=0) for t in textures], axis=0
                                    )
                                ),
                            )
                            for slot, textures in sorted(by_slot.items())
                        )
                        face_thumbnails = tuple(
                            (slot, _encode_thumbnail(textures[0]))
                            for slot, textures in sorted(by_slot.items())
                        )
                        overall = tuple(
                            int(round(value))
                            for value in np.mean(
                                [face.texture.reshape(-1, 3).mean(axis=0) for face in textured],
                                axis=0,
                            )
                        )
                        samples.append(
                            TextureSample(
                                BlockState(
                                    f"minecraft:{block_name}",
                                    _complete_properties(block_name, properties),
                                ),
                                overall,
                                "",
                                (),
                                (overall,),
                                _model_family(block_name, []),
                                (),
                                False,
                                face_map,
                                face_thumbnails,
                                (),
                                tuple(
                                    (face.slot,
                                     tuple(tuple(float(v) for v in corner) for corner in face.corners),
                                     tuple(float(v) for v in face.normal), face.cullface,
                                     _encode_alpha_mask(face.alpha))
                                    for face in shape.faces
                                ),
                                shape.boxes,
                                tuple(slot for slot in FACE_SLOTS if shape.covers(slot)),
                            )
                        )

        self.samples = _deduplicate_samples(samples)
        self.jar_path = jar_path.resolve()
        self.source = "extracted"
        _save_cached_samples(cache_path, jar_path, self.samples)
        return len(self.samples)

    def palette(self) -> tuple[PaletteColor, ...]:
        return tuple(_palette_color(sample) for sample in self.samples)

    def palette_for_states(
        self, states: Iterable[BlockState], *, include_partial: bool = True
    ) -> tuple[PaletteColor, ...]:
        """Return model-aware entries for the requested exact block states.

        The canonical and complete palettes intentionally de-duplicate entries
        that look alike. That is useful for an unconstrained classifier, but it
        is wrong when a reference or an accepted edit names a rare state: the
        renderer needs that state's baked faces, shape boxes, and alpha masks.
        Keep the requested states distinct and preserve their full geometry. Set
        ``include_partial=False`` when the result is the first-pass material
        vocabulary; partial models should then enter only a shape-gated pass.
        """
        requested = tuple(states)
        if not requested:
            return ()
        exact = {sample.state: sample for sample in self.samples}
        selected: list[PaletteColor] = []
        for state in requested:
            # Blockstate extraction materialises only defaults that are justified
            # by the block family. Resolve against that normalized key, never by
            # name alone: a missing top slab, stair facing, or fence connection
            # must remain unavailable rather than inherit another variant's mesh.
            normalized = BlockState(
                state.name, _complete_properties(state.name, state.properties)
            )
            sample = exact.get(state) or exact.get(normalized)
            if sample is not None and (include_partial or sample.full_cube):
                # Keep the caller's state key so calibration and renderer lookups
                # line up even when the caller omitted a justified default.
                selected.append(_palette_color(sample, state=state))
        return tuple(selected)

    def canonical_palette(self, *, tolerance: int = COLOUR_RESOLUTION) -> tuple[PaletteColor, ...]:
        """Full-cube states, one per distinct world-oriented appearance.

        Any block whose model is a full cube is a candidate. Membership is decided
        by geometry alone: an earlier version also required the block to be in the
        "cube" or "log" family, which silently dropped 389 real full-cube blocks,
        ``stone``, ``blackstone`` and ``basalt`` among them, because they carry
        more than one texture and no recognised name token.

        Rotated variants of the same model (a log's three axes, say) differ in
        which world face carries which texture, so they survive de-duplication
        as separate entries. States whose faces all agree to within ``tolerance``
        colour levels are indistinguishable to a colour classifier, so only one
        of them is kept, chosen by :func:`_canonical_rank`.
        """
        best: dict[tuple[int, ...], TextureSample] = {}
        order: list[tuple[int, ...]] = []
        for sample in self.samples:
            if not sample.full_cube or _is_biome_tinted(sample.state.name):
                continue
            appearance = _appearance_key(sample, tolerance)
            previous = best.get(appearance)
            if previous is None:
                best[appearance] = sample
                order.append(appearance)
            elif _canonical_rank(sample) < _canonical_rank(previous):
                best[appearance] = sample
        return tuple(_palette_color(best[appearance]) for appearance in order)

    def complete_palette(
        self, *, tolerance: int = COLOUR_RESOLUTION, include_tinted: bool = True
    ) -> tuple[PaletteColor, ...]:
        """Every block the game can place, full cubes and partial shapes alike.

        `canonical_palette` keeps only full cubes, which is why a reconstruction
        contained exactly zero stairs, slabs, carpets, fences and torches - not
        because the classifier judged them badly, but because its vocabulary did
        not contain them. Template 2's reference is 8.1% partial blocks and no
        amount of classifier work could ever have reached them.

        Shape belongs in the de-duplication key alongside appearance. `oak_stairs`
        and `oak_slab` are cut from the same plank texture and so share an
        appearance key exactly; keyed on looks alone one of them silently replaces
        the other. Keyed on both, every distinct thing a camera could see survives.

        No volume floor. `partial_palette` drops shapes below a hundredth of a cell
        on the grounds that a lever projects to too few pixels to judge, which is
        true and is a matter for the classifier's evidence weighting rather than
        for the vocabulary: a block that cannot be named can never be recovered,
        and the palette is not the place to decide a block is not worth trying.
        """
        best: dict[tuple, TextureSample] = {}
        order: list[tuple] = []
        for sample in self.samples:
            if not include_tinted and _is_biome_tinted(sample.state.name):
                continue
            key = (_shape_key(sample), _appearance_key(sample, tolerance))
            previous = best.get(key)
            if previous is None:
                best[key] = sample
                order.append(key)
            elif _canonical_rank(sample) < _canonical_rank(previous):
                best[key] = sample
        return tuple(_palette_color(best[key]) for key in order)

    def partial_palette(
        self, *, tolerance: int = COLOUR_RESOLUTION, minimum_volume: float = 0.008
    ) -> tuple[TextureSample, ...]:
        """States whose model is not a full cube, one per distinct shape and look.

        The full-cube palette deliberately excludes these, because the classifier
        compares complete-cube appearances. They are still real blocks that a
        camera can see, and nothing else in the pipeline proposes their cells, so
        they need a candidate list of their own.

        ``minimum_volume`` drops shapes too slight to leave enough pixels to judge
        - a lever or a piece of string projects to almost nothing, and admitting
        such a candidate on a handful of pixels is guessing rather than measuring.
        The floor sits just under a torch, which occupies about a hundredth of its
        cell and is common enough in real builds to be worth the risk.
        """
        best: dict[tuple, TextureSample] = {}
        order: list[tuple] = []
        for sample in self.samples:
            if sample.full_cube or _is_biome_tinted(sample.state.name):
                continue
            if not sample.shape_boxes:
                continue
            volume = sum(
                max(0.0, high[0] - low[0]) * max(0.0, high[1] - low[1]) * max(0.0, high[2] - low[2])
                for low, high in sample.shape_boxes
            )
            if volume < minimum_volume:
                continue
            # Shape and appearance both matter: two carpets differ only in colour,
            # a slab and its upside-down twin only in shape.
            shape = tuple(
                (round(v, 3) for v in low) for low, _high in sorted(sample.shape_boxes)
            )
            shape = tuple(
                (tuple(round(v, 3) for v in low), tuple(round(v, 3) for v in high))
                for low, high in sorted(sample.shape_boxes)
            )
            key = (shape, _appearance_key(sample, tolerance))
            if key not in best:
                best[key] = sample
                order.append(key)
            elif _canonical_rank(sample) < _canonical_rank(best[key]):
                best[key] = sample
        return tuple(best[key] for key in order)

    def resolve(self, block_name: str) -> TextureSample | None:
        return next((sample for sample in self.samples if sample.state.name == block_name), None)


# Blocks the game recolours per biome. Their texture in the jar is a grey or
# washed-out mask that the renderer tints, so the palette colour is not what a
# screenshot shows, and leaving them in lets a grey leaf match any grey block.
BIOME_TINTED = (
    "leaves", "grass_block", "vine", "lily_pad", "sugar_cane", "fern", "tall_grass",
    "water", "bubble_column", "spawner", "attached_melon", "attached_pumpkin",
    "redstone_wire", "mangrove_roots",
)


def _is_biome_tinted(name: str) -> bool:
    plain = name.rsplit(":", 1)[-1]
    return any(token in plain for token in BIOME_TINTED)


def _canonical_rank(sample: TextureSample) -> tuple[int, int, str]:
    """Order candidates for an appearance group so the plainest block wins.

    Derived blocks name themselves after the base one and add to it -
    ``infested_stone``, ``cracked_deepslate_bricks``, ``waxed_exposed_copper`` -
    so among blocks that look identical the shortest name is the base material.
    Without this, ``stone`` lost its own appearance group to ``infested_stone``
    on an alphabetical tie-break.
    """
    return (len(sample.state.properties), len(sample.state.name), sample.state.name)


def _appearance_key(
    sample: TextureSample, tolerance: int, pattern_levels: int = PATTERN_LEVELS
) -> tuple[int, ...]:
    """Quantised six-face signature: equal keys mean indistinguishable.

    Mean colour alone is not enough. ``cobblestone`` and ``stone`` average to the
    same grey and so collapsed into one candidate, which left cobblestone with no
    entry able to win its own blocks; the same holds for any pair that differs in
    pattern rather than in tint. The classifier compares whole rectified textures
    now, so it can tell a lumpy face from a smooth one - but only if both survive
    into the palette. Folding a coarse description of each face's pattern into the
    key keeps them apart, while still merging states that really do look alike.
    """
    faces = dict(sample.face_map)
    thumbnails = dict(sample.face_thumbnails)
    step = max(1, tolerance)
    values: list[int] = []
    for slot in FACE_SLOTS:
        color = faces.get(slot) or (sample.face_rgbs[0] if sample.face_rgbs else sample.rgb)
        values.extend(int(round(channel / step)) for channel in color)
        values.extend(_pattern_signature(thumbnails.get(slot), pattern_levels))
    return tuple(values)


def _pattern_signature(encoded: str | None, levels: int) -> tuple[int, ...]:
    """A coarse, brightness-independent description of one face's pattern."""
    if not encoded or levels <= 0:
        return ()
    import base64
    from io import BytesIO

    try:
        from PIL import Image

        pixels = np.asarray(Image.open(BytesIO(base64.b64decode(encoded))), dtype=np.float32)
    except (ValueError, OSError, TypeError, ImportError):
        return ()
    if pixels.ndim != 3:
        return ()
    luminance = pixels.mean(axis=2)
    spread = float(luminance.std())
    if spread < 1e-3:
        return (0,) * (PATTERN_CELLS * PATTERN_CELLS + 1)
    # Mean-pool to a small grid and normalise, so the signature describes how the
    # face's light and dark areas are arranged rather than how bright it is.
    side = luminance.shape[0] // PATTERN_CELLS
    if side < 1:
        return (0,) * (PATTERN_CELLS * PATTERN_CELLS + 1)
    pooled = luminance[: side * PATTERN_CELLS, : side * PATTERN_CELLS]
    pooled = pooled.reshape(PATTERN_CELLS, side, PATTERN_CELLS, side).mean(axis=(1, 3))
    scaled = (pooled - pooled.mean()) / spread
    signature = [int(round(value * levels)) for value in scaled.reshape(-1)]
    signature.append(int(round(spread / max(PATTERN_SPREAD_STEP, 1e-6))))
    return tuple(signature)


def _shape_key(sample: TextureSample) -> tuple:
    """The model's occupied volume, quantised, or ``None`` for a full cube."""
    if sample.full_cube:
        return ()
    return tuple(
        (tuple(round(v, 3) for v in low), tuple(round(v, 3) for v in high))
        for low, high in sorted(sample.shape_boxes)
    )


def _palette_color(sample: TextureSample, *, state: BlockState | None = None) -> PaletteColor:
    return PaletteColor(
        sample.state if state is None else state,
        sample.rgb,
        sample.face_rgbs,
        sample.family,
        sample.descriptors,
        sample.face_map,
        sample.face_thumbnails,
        sample.face_corners,
        sample.model_faces,
        sample.full_cube,
        sample.shape_boxes,
        sample.covered_slots,
    )


def _model_variants(blockstate: object) -> list[tuple[tuple[tuple[str, str], ...], str]]:
    if not isinstance(blockstate, dict):
        return []
    values: list[object] = []
    variants = blockstate.get("variants")
    if isinstance(variants, dict):
        values.extend(variants.values())
    multipart = blockstate.get("multipart")
    if isinstance(multipart, list):
        values.extend(part.get("apply") for part in multipart if isinstance(part, dict))
    variants: list[tuple[tuple[tuple[str, str], ...], str]] = []
    raw_variants = blockstate.get("variants") if isinstance(blockstate, dict) else None
    for key, value in raw_variants.items() if isinstance(raw_variants, dict) else [("", value) for value in values]:
        entries = value if isinstance(value, list) else [value]
        for entry in entries:
            if isinstance(entry, dict) and isinstance(entry.get("model"), str):
                properties = tuple(
                    sorted(
                        (part.split("=", 1)[0], part.split("=", 1)[1])
                        for part in str(key).split(",")
                        if "=" in part
                    )
                )
                variants.append((properties, entry["model"]))
    return variants


def _resolve_model_textures(archive: zipfile.ZipFile, model_name: str, seen: set[str]) -> list[str]:
    if model_name in seen:
        return []
    seen.add(model_name)
    path = f"assets/minecraft/models/{model_name.removeprefix('minecraft:')}.json"
    try:
        model = json.loads(archive.read(path))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    parent = model.get("parent") if isinstance(model, dict) else None
    inherited = _resolve_model_texture_map(archive, parent, seen) if isinstance(parent, str) else {}
    textures = model.get("textures", {}) if isinstance(model, dict) else {}
    if isinstance(textures, dict):
        inherited.update({key: value for key, value in textures.items() if isinstance(value, str)})
    resolved: list[str] = []
    for value in inherited.values():
        resolved_name = value
        visited: set[str] = set()
        while resolved_name.startswith("#") and resolved_name[1:] not in visited:
            visited.add(resolved_name[1:])
            resolved_name = inherited.get(resolved_name[1:], resolved_name)
        if isinstance(resolved_name, str) and not resolved_name.startswith("#"):
            resolved.append(f"assets/minecraft/textures/{resolved_name.removeprefix('minecraft:')}.png")
    return list(dict.fromkeys(resolved))


def _resolve_face_textures(archive: zipfile.ZipFile, model_name: str, seen: set[str]) -> tuple[str, ...]:
    if model_name in seen:
        return ()
    seen.add(model_name)
    path = f"assets/minecraft/models/{model_name.removeprefix('minecraft:')}.json"
    try:
        model = json.loads(archive.read(path))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError):
        return ()
    if not isinstance(model, dict):
        return ()

    inherited = _resolve_model_texture_map(archive, model.get("parent"), seen) if isinstance(model.get("parent"), str) else {}
    textures = model.get("textures", {}) if isinstance(model.get("textures"), dict) else {}
    if isinstance(textures, dict):
        inherited.update({key: value for key, value in textures.items() if isinstance(value, str)})

    face_slots = ("up", "down", "north", "south", "east", "west")
    ordered: list[str] = []
    for slot in face_slots:
        face_textures = model.get("elements", [])
        matched = None
        for element in face_textures if isinstance(face_textures, list) else []:
            if not isinstance(element, dict):
                continue
            faces = element.get("faces")
            if not isinstance(faces, dict):
                continue
            face = faces.get(slot)
            if not isinstance(face, dict):
                continue
            texture_ref = face.get("texture")
            if not isinstance(texture_ref, str):
                continue
            matched = texture_ref
            break
        if matched is None:
            if slot in inherited:
                matched = inherited[slot]
            elif slot == "up" and "all" in inherited:
                matched = inherited["all"]
            elif slot == "down" and "all" in inherited:
                matched = inherited["all"]
            elif slot in {"north", "south", "east", "west"} and "side" in inherited:
                matched = inherited["side"]
        if matched is None:
            continue
        resolved_name = matched
        visited: set[str] = set()
        while resolved_name.startswith("#") and resolved_name[1:] not in visited:
            visited.add(resolved_name[1:])
            resolved_name = inherited.get(resolved_name[1:], resolved_name)
        if isinstance(resolved_name, str) and not resolved_name.startswith("#"):
            ordered.append(f"assets/minecraft/textures/{resolved_name.removeprefix('minecraft:')}.png")

    return tuple(dict.fromkeys(ordered))


def _resolve_face_texture_slots(archive: zipfile.ZipFile, model_name: str, seen: set[str]) -> dict[str, str]:
    if model_name in seen:
        return {}
    seen.add(model_name)
    path = f"assets/minecraft/models/{model_name.removeprefix('minecraft:')}.json"
    try:
        model = json.loads(archive.read(path))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    if not isinstance(model, dict):
        return {}
    parent = model.get("parent")
    inherited = _resolve_face_texture_slots(archive, parent, seen) if isinstance(parent, str) else {}
    texture_map = _resolve_model_texture_map(archive, parent, set()) if isinstance(parent, str) else {}
    textures = model.get("textures", {}) if isinstance(model.get("textures"), dict) else {}
    if isinstance(textures, dict):
        texture_map.update({key: value for key, value in textures.items() if isinstance(value, str)})
    slots = ("up", "down", "north", "south", "east", "west")
    for slot in slots:
        matched = None
        elements = model.get("elements", [])
        for element in elements if isinstance(elements, list) else []:
            faces = element.get("faces") if isinstance(element, dict) else None
            face = faces.get(slot) if isinstance(faces, dict) else None
            if isinstance(face, dict) and isinstance(face.get("texture"), str):
                matched = face["texture"]
                break
        if matched is None:
            matched = texture_map.get(slot) or texture_map.get("all") if slot in {"up", "down"} else texture_map.get(slot) or texture_map.get("side")
        if not isinstance(matched, str):
            continue
        resolved = matched
        visited: set[str] = set()
        while resolved.startswith("#") and resolved[1:] not in visited:
            visited.add(resolved[1:])
            resolved = texture_map.get(resolved[1:], resolved)
        if not resolved.startswith("#"):
            inherited[slot] = f"assets/minecraft/textures/{resolved.removeprefix('minecraft:')}.png"
    return inherited


def _model_has_full_cube_geometry(archive: zipfile.ZipFile, model_name: str, seen: set[str]) -> bool:
    if model_name in seen:
        return False
    seen.add(model_name)
    path = f"assets/minecraft/models/{model_name.removeprefix('minecraft:')}.json"
    try:
        model = json.loads(archive.read(path))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(model, dict):
        return False
    elements = model.get("elements")
    if isinstance(elements, list):
        for element in elements:
            if not isinstance(element, dict) or element.get("from") != [0, 0, 0] or element.get("to") != [16, 16, 16]:
                continue
            faces = element.get("faces")
            if isinstance(faces, dict) and all(face in faces for face in ("up", "down", "north", "south", "east", "west")):
                return True
    parent = model.get("parent")
    return isinstance(parent, str) and _model_has_full_cube_geometry(archive, parent, seen)


def _resolve_model_texture_map(
    archive: zipfile.ZipFile, model_name: str, seen: set[str]
) -> dict[str, str]:
    if model_name in seen:
        return {}
    seen.add(model_name)
    path = f"assets/minecraft/models/{model_name.removeprefix('minecraft:')}.json"
    try:
        model = json.loads(archive.read(path))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    parent = model.get("parent") if isinstance(model, dict) else None
    values = _resolve_model_texture_map(archive, parent, seen) if isinstance(parent, str) else {}
    textures = model.get("textures", {}) if isinstance(model, dict) else {}
    if isinstance(textures, dict):
        values.update({key: value for key, value in textures.items() if isinstance(value, str)})
    return values


def _model_family(block_name: str, texture_paths: list[str]) -> str:
    name = block_name.lower()
    if "stairs" in name:
        return "stairs"
    if "slab" in name:
        return "slab"
    if "fence" in name or "wall" in name or "gate" in name:
        return "fence"
    if "carpet" in name:
        return "carpet"
    if any(token in name for token in ("sapling", "flower", "grass", "bush", "fern", "mushroom")):
        return "plant"
    if any(token in name for token in ("log", "wood", "planks")):
        return "log"
    if not texture_paths or any(token in name for token in ("torch", "lantern", "rail", "button", "pressure_plate")):
        return "utility"
    if "cube" in name or len(set(texture_paths)) == 1:
        return "cube"
    return "other"


def _complete_properties(
    block_name: str, properties: tuple[tuple[str, str], ...]
) -> tuple[tuple[str, str], ...]:
    """Materialize common Minecraft defaults omitted by blockstate variant keys."""
    completed = dict(properties)
    name = block_name.lower()
    if "stairs" in name:
        completed.setdefault("facing", "north")
        completed.setdefault("half", "bottom")
        completed.setdefault("shape", "straight")
        completed.setdefault("waterlogged", "false")
    elif "slab" in name:
        completed.setdefault("type", "bottom")
        completed.setdefault("waterlogged", "false")
    elif "fence" in name:
        for direction in ("east", "north", "south", "west"):
            completed.setdefault(direction, "false")
        completed.setdefault("waterlogged", "false")
    elif name.endswith("_log") or name.endswith("_wood"):
        completed.setdefault("axis", "y")
    return tuple(sorted(completed.items()))

def _deduplicate_samples(samples: list[TextureSample]) -> list[TextureSample]:
    seen: set[BlockState] = set()
    result: list[TextureSample] = []
    for sample in samples:
        if sample.state not in seen:
            seen.add(sample.state)
            result.append(sample)
    return result


def _texture_descriptor(rgba: np.ndarray, alpha: np.ndarray) -> tuple[float, ...]:
    rgb = rgba[:, :, :3]
    valid = alpha > 0
    pixels = rgb[valid]
    if len(pixels) == 0:
        return (0.0,) * 7
    normalized = pixels / np.maximum(pixels.sum(axis=1, keepdims=True), 1.0)
    histogram = np.histogram(normalized[:, 0], bins=4, range=(0.0, 1.0), density=True)[0]
    luminance = pixels.mean(axis=1)
    gradients = np.abs(np.diff(rgb.mean(axis=2), axis=0)).mean()
    return tuple(float(value) for value in (*histogram, luminance.mean() / 255.0, luminance.std() / 255.0, gradients / 255.0))


def _thumbnail_pixels(rgba: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgba[:, :, :3], 0.0, 255.0).astype(np.uint8)
    if rgb.shape[:2] == (16, 16):
        return rgb
    from PIL import Image

    return np.asarray(Image.fromarray(rgb, mode="RGB").resize((16, 16), Image.Resampling.BILINEAR))


def _encode_thumbnail(pixels: np.ndarray) -> str:
    from PIL import Image

    buffer = BytesIO()
    Image.fromarray(pixels.astype(np.uint8), mode="RGB").save(buffer, format="PNG", compress_level=9)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _encode_alpha_mask(alpha: np.ndarray | None) -> str | None:
    """Persist a face's baked cut-out mask compactly alongside its polygon."""
    if alpha is None:
        return None
    from PIL import Image

    buffer = BytesIO()
    Image.fromarray(np.clip(alpha * 255.0, 0, 255).astype(np.uint8), mode="L").save(
        buffer, format="PNG", compress_level=9
    )
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _load_cached_samples(cache_path: Path, jar_path: Path) -> list[TextureSample] | None:
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("cache_schema") != CACHE_SCHEMA_VERSION:
        return None
    if data.get("jar_size") != jar_path.stat().st_size or data.get("jar_mtime") != jar_path.stat().st_mtime_ns:
        return None
    samples = data.get("samples", [])
    if not isinstance(samples, list) or not samples:
        return None
    parsed: list[TextureSample] = []
    for item in samples:
        state = BlockState.from_dict(item["state"])
        parsed.append(
            TextureSample(
                state,
                tuple(item["rgb"]),
                item["texture_path"],
                tuple(item.get("face_paths", [])),
                tuple(tuple(color) for color in item.get("face_rgbs", [])),
                item.get("family", "other"),
                tuple(tuple(value) for value in item.get("descriptors", [])),
                bool(item.get("full_cube", False)),
                tuple((str(slot), tuple(color)) for slot, color in item.get("face_map", [])),
                tuple((str(slot), str(encoded)) for slot, encoded in item.get("face_thumbnails", [])),
                tuple(
                    (str(slot), tuple(tuple(float(v) for v in corner) for corner in corners))
                    for slot, corners in item.get("face_corners", [])
                ),
                tuple(
                    (
                        str(record[0]),
                        tuple(tuple(float(v) for v in corner) for corner in record[1]),
                        tuple(float(v) for v in record[2]),
                        str(record[3]) if record[3] is not None else None,
                        str(record[4]) if len(record) > 4 and record[4] is not None else None,
                    )
                    for record in item.get("model_faces", [])
                    if isinstance(record, list) and len(record) >= 4
                ),
                tuple(
                    (
                        tuple(float(value) for value in box[0]),
                        tuple(float(value) for value in box[1]),
                    )
                    for box in item.get("shape_boxes", [])
                ),
                tuple(str(slot) for slot in item.get("covered_slots", [])),
            )
        )
    return parsed


def _save_cached_samples(cache_path: Path, jar_path: Path, samples: list[TextureSample]) -> None:
    data = {
        "cache_schema": CACHE_SCHEMA_VERSION,
        "jar_size": jar_path.stat().st_size,
        "jar_mtime": jar_path.stat().st_mtime_ns,
        "samples": [
            {
                "state": sample.state.to_dict(),
                "rgb": sample.rgb,
                "texture_path": sample.texture_path,
                "face_paths": sample.face_paths,
                "face_rgbs": sample.face_rgbs,
                "family": sample.family,
                "descriptors": sample.descriptors,
                "full_cube": sample.full_cube,
                "face_map": sample.face_map,
                "face_thumbnails": sample.face_thumbnails,
                "face_corners": sample.face_corners,
                "model_faces": sample.model_faces,
                "shape_boxes": sample.shape_boxes,
                "covered_slots": sample.covered_slots,
            }
            for sample in samples
        ],
    }
    try:
        cache_path.write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        pass
