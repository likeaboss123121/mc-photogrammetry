"""Bake vanilla block models into world-oriented per-face appearances.

A blockstate variant may rotate its model (``"x": 90, "y": 90`` on a horizontal
log, for example). The rotation decides which world face shows the end grain and
which shows the bark, so a palette that ignores it can never distinguish
``axis=x`` from ``axis=z``. Baking resolves each variant into the six world face
slots, each carrying its texture and the block-local corners of that face in the
order the texture's UV axes run.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

FACE_SLOTS = ("down", "up", "north", "south", "west", "east")

# Per model-space face: (normal, uv origin, +u edge, +v edge) as block-local
# offsets from the block centre, derived from vanilla's automatic UVs for a
# 0..16 cube. The origin is the corner carrying texture pixel (0, 0).
FACE_BASIS: dict[str, tuple[tuple[float, ...], ...]] = {
    "up": ((0, 1, 0), (-0.5, 0.5, -0.5), (1, 0, 0), (0, 0, 1)),
    "down": ((0, -1, 0), (-0.5, -0.5, 0.5), (1, 0, 0), (0, 0, -1)),
    "north": ((0, 0, -1), (0.5, 0.5, -0.5), (-1, 0, 0), (0, -1, 0)),
    "south": ((0, 0, 1), (-0.5, 0.5, 0.5), (1, 0, 0), (0, -1, 0)),
    "west": ((-1, 0, 0), (-0.5, 0.5, -0.5), (0, 0, 1), (0, -1, 0)),
    "east": ((1, 0, 0), (0.5, 0.5, 0.5), (0, 0, -1), (0, -1, 0)),
}

TEXTURE_SIZE = 16

FACE_SHADE = {"up": 1.0, "down": 0.5, "north": 0.8, "south": 0.8, "east": 0.6, "west": 0.6}
"""Vanilla's fixed per-direction brightness, applied before any block lighting."""


@dataclass(frozen=True, slots=True)
class FaceAppearance:
    """One world-facing face of a baked block model."""

    slot: str
    texture: np.ndarray
    """``(16, 16, 3)`` float32 RGB, oriented so row/column follow the face's UV axes."""
    corners: np.ndarray
    """``(4, 3)`` block-local corner offsets matching texture corners (0,0), (u,0), (u,v), (0,v)."""

    @property
    def rgb(self) -> tuple[int, int, int]:
        return tuple(int(round(value)) for value in self.texture.reshape(-1, 3).mean(axis=0))


def _rotation_x(degrees: float) -> np.ndarray:
    radians = np.radians(degrees)
    cos, sin = np.cos(radians), np.sin(radians)
    return np.asarray([[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]])


def _rotation_y(degrees: float) -> np.ndarray:
    radians = np.radians(degrees)
    cos, sin = np.cos(radians), np.sin(radians)
    return np.asarray([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]])


def _rotation_z(degrees: float) -> np.ndarray:
    radians = np.radians(degrees)
    cos, sin = np.cos(radians), np.sin(radians)
    return np.asarray([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


def variant_rotation(x_degrees: float, y_degrees: float) -> np.ndarray:
    """Vanilla ``BlockModelRotation``: rotate by ``-x`` about +X, then ``-y`` about +Y."""
    return _rotation_y(-y_degrees) @ _rotation_x(-x_degrees)


def _slot_for_normal(normal: np.ndarray) -> str | None:
    for slot, (reference, *_rest) in FACE_BASIS.items():
        if np.allclose(normal, np.asarray(reference, dtype=float), atol=1e-6):
            return slot
    return None


def _nearest_slot_for_normal(normal: np.ndarray) -> str:
    """Cardinal ownership bucket for a non-cardinal face; keep its real normal too."""
    return max(FACE_SLOTS, key=lambda slot: float(np.asarray(FACE_BASIS[slot][0]) @ normal))


def element_rotation(element: dict) -> tuple[np.ndarray, np.ndarray]:
    """Vanilla element-local rotation matrix and pivot in centred block coordinates."""
    raw = element.get("rotation")
    if not isinstance(raw, dict):
        return np.eye(3), np.zeros(3)
    makers = {"x": _rotation_x, "y": _rotation_y, "z": _rotation_z}
    maker = makers.get(raw.get("axis"))
    if maker is None:
        return np.eye(3), np.zeros(3)
    matrix = maker(float(raw.get("angle", 0.0)))
    origin = raw.get("origin", [8, 8, 8])
    try:
        pivot = np.asarray(origin, dtype=float) / 16.0 - 0.5
    except (TypeError, ValueError):
        pivot = np.zeros(3)
    return matrix, pivot


def blockstate_variants(blockstate: object) -> list[tuple[tuple[tuple[str, str], ...], str, float, float]]:
    """Yield ``(properties, model, x_degrees, y_degrees)`` for each blockstate variant."""
    if not isinstance(blockstate, dict):
        return []
    raw = blockstate.get("variants")
    if not isinstance(raw, dict):
        return []
    variants = []
    for key, value in raw.items():
        entry = value[0] if isinstance(value, list) and value else value
        if not isinstance(entry, dict) or not isinstance(entry.get("model"), str):
            continue
        properties = tuple(
            sorted(
                (part.split("=", 1)[0], part.split("=", 1)[1])
                for part in str(key).split(",")
                if "=" in part
            )
        )
        variants.append((properties, entry["model"], float(entry.get("x", 0)), float(entry.get("y", 0))))
    return variants


class ModelAtlas:
    """Resolve and bake vanilla block models out of an open client jar."""

    def __init__(self, archive: zipfile.ZipFile) -> None:
        self._archive = archive
        self._models: dict[str, dict] = {}
        self._textures: dict[str, np.ndarray | None] = {}

    def _model(self, name: str) -> dict:
        key = name.removeprefix("minecraft:")
        if key not in self._models:
            try:
                loaded = json.loads(self._archive.read(f"assets/minecraft/models/{key}.json"))
            except (KeyError, ValueError, UnicodeDecodeError):
                loaded = {}
            self._models[key] = loaded if isinstance(loaded, dict) else {}
        return self._models[key]

    def texture(self, reference: str) -> np.ndarray | None:
        """Load a block texture as ``(16, 16, 3)`` float32, or ``None`` when unavailable."""
        key = reference.removeprefix("minecraft:")
        if key in self._textures:
            return self._textures[key]
        from PIL import Image

        try:
            with Image.open(self._archive.open(f"assets/minecraft/textures/{key}.png")) as image:
                rgba = np.asarray(image.convert("RGBA"), dtype=np.float32)
        except (KeyError, OSError, ValueError):
            self._textures[key] = None
            return None
        height, width = rgba.shape[:2]
        if height > width and width > 0 and height % width == 0:
            rgba = rgba[:width]  # animated strip: keep the first frame
        rgb, alpha = rgba[:, :, :3], rgba[:, :, 3:4] / 255.0
        weight = float(alpha.sum())
        if weight <= 0.0:
            self._textures[key] = None
            return None
        # Composite over the texture's own mean so cut-outs do not read as black.
        mean = (rgb * alpha).sum(axis=(0, 1)) / weight
        flat = rgb * alpha + mean * (1.0 - alpha)
        if flat.shape[:2] != (TEXTURE_SIZE, TEXTURE_SIZE):
            flat = np.asarray(
                Image.fromarray(np.clip(flat, 0, 255).astype(np.uint8)).resize(
                    (TEXTURE_SIZE, TEXTURE_SIZE), Image.Resampling.BILINEAR
                ),
                dtype=np.float32,
            )
        self._textures[key] = np.ascontiguousarray(flat, dtype=np.float32)
        return self._textures[key]

    def _flatten(self, model_name: str) -> tuple[dict[str, str], list | None]:
        """Merge the parent chain into one texture map plus the nearest element list."""
        chain: list[dict] = []
        name: str | None = model_name
        while name and len(chain) < 16:
            model = self._model(name)
            chain.append(model)
            parent = model.get("parent")
            name = parent if isinstance(parent, str) else None
        texture_map: dict[str, str] = {}
        for model in reversed(chain):
            textures = model.get("textures")
            if isinstance(textures, dict):
                texture_map.update({k: v for k, v in textures.items() if isinstance(v, str)})
        elements = next(
            (model["elements"] for model in chain if isinstance(model.get("elements"), list)), None
        )
        return texture_map, elements

    @staticmethod
    def _resolve(texture_map: dict[str, str], reference: object) -> str | None:
        seen: set[str] = set()
        while isinstance(reference, str) and reference.startswith("#"):
            key = reference[1:]
            if key in seen:
                return None
            seen.add(key)
            reference = texture_map.get(key)
        return reference if isinstance(reference, str) else None

    def bake(self, model_name: str, x_degrees: float = 0.0, y_degrees: float = 0.0) -> dict[str, FaceAppearance] | None:
        """Bake a full-cube model into its six world face appearances, or ``None``."""
        texture_map, elements = self._flatten(model_name)
        if not isinstance(elements, list):
            return None
        cube = next(
            (
                element
                for element in elements
                if isinstance(element, dict)
                and element.get("from") == [0, 0, 0]
                and element.get("to") == [16, 16, 16]
                and isinstance(element.get("faces"), dict)
                and all(slot in element["faces"] for slot in FACE_SLOTS)
            ),
            None,
        )
        if cube is None:
            return None
        rotation = variant_rotation(x_degrees, y_degrees)
        baked: dict[str, FaceAppearance] = {}
        for slot in FACE_SLOTS:
            face = cube["faces"][slot]
            reference = self._resolve(texture_map, face.get("texture") if isinstance(face, dict) else None)
            if reference is None:
                return None
            texture = self.texture(reference)
            if texture is None:
                return None
            turns = int(face.get("rotation", 0)) // 90 % 4 if isinstance(face, dict) else 0
            if turns:
                texture = np.ascontiguousarray(np.rot90(texture, k=turns))
            normal, origin, edge_u, edge_v = FACE_BASIS[slot]
            world_normal = rotation @ np.asarray(normal, dtype=float)
            world_slot = _slot_for_normal(world_normal)
            if world_slot is None:
                return None
            origin = rotation @ np.asarray(origin, dtype=float)
            edge_u = rotation @ np.asarray(edge_u, dtype=float)
            edge_v = rotation @ np.asarray(edge_v, dtype=float)
            corners = np.asarray([origin, origin + edge_u, origin + edge_u + edge_v, origin + edge_v])
            baked[world_slot] = FaceAppearance(world_slot, texture, corners)
        return baked if len(baked) == len(FACE_SLOTS) else None

    def _element_face_texture(self, texture_map: dict[str, str], face: dict) -> np.ndarray | None:
        """The texture for one element face, cropped to its UV window."""
        reference = self._resolve(texture_map, face.get("texture"))
        if reference is None:
            return None
        texture = self.texture(reference)
        if texture is None:
            return None
        uv = face.get("uv")
        if isinstance(uv, list) and len(uv) == 4:
            # A slab's side is the lower half of the block's texture, not the
            # whole of it squashed; without the crop a partial face would carry
            # the wrong pattern entirely.
            from PIL import Image

            left, top, right, bottom = (float(value) * TEXTURE_SIZE / 16.0 for value in uv)
            x0, x1 = sorted((int(round(left)), int(round(right))))
            y0, y1 = sorted((int(round(top)), int(round(bottom))))
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(TEXTURE_SIZE, max(x1, x0 + 1)), min(TEXTURE_SIZE, max(y1, y0 + 1))
            window = texture[y0:y1, x0:x1]
            if window.size and window.shape[:2] != (TEXTURE_SIZE, TEXTURE_SIZE):
                texture = np.asarray(
                    Image.fromarray(np.clip(window, 0, 255).astype(np.uint8)).resize(
                        (TEXTURE_SIZE, TEXTURE_SIZE), Image.Resampling.BILINEAR
                    ),
                    dtype=np.float32,
                )
            elif window.size:
                texture = window
        turns = int(face.get("rotation", 0)) // 90 % 4
        if turns:
            texture = np.ascontiguousarray(np.rot90(texture, k=turns))
        return np.ascontiguousarray(texture, dtype=np.float32)

    def geometry(
        self,
        model_name: str,
        x_degrees: float = 0.0,
        y_degrees: float = 0.0,
        *,
        include_textures: bool = True,
    ) -> "ModelGeometry | None":
        """Bake every element of a model, whatever shape it is.

        Unlike ``bake`` this does not require a complete cube, so slabs, carpets
        and stairs come back with real geometry instead of being refused.

        Set ``include_textures`` false when only the shape is wanted. Resolving
        and cropping a texture for every face of every variant in the jar costs
        far more than reading the boxes, and occlusion needs only the boxes.
        """
        texture_map, elements = self._flatten(model_name)
        if not isinstance(elements, list) or not elements:
            return None
        rotation = variant_rotation(x_degrees, y_degrees)
        faces: list[ElementFace] = []
        rotated_corners: list[np.ndarray] = []
        element_boxes: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
        for element in elements:
            if not isinstance(element, dict):
                continue
            start, end = element.get("from"), element.get("to")
            if not (isinstance(start, list) and isinstance(end, list)):
                continue
            if len(start) != 3 or len(end) != 3:
                continue
            try:
                low = np.minimum(np.asarray(start, dtype=float), np.asarray(end, dtype=float)) / 16.0
                high = np.maximum(np.asarray(start, dtype=float), np.asarray(end, dtype=float)) / 16.0
            except (TypeError, ValueError):
                continue
            box = np.asarray(
                [[low[0] if a else high[0], low[1] if b else high[1], low[2] if c else high[2]]
                 for a in (0, 1) for b in (0, 1) for c in (0, 1)],
                dtype=float,
            ) - 0.5
            local_rotation, pivot = element_rotation(element)
            turned = ((box - pivot) @ local_rotation.T + pivot) @ rotation.T
            rotated_corners.append(turned)
            element_boxes.append((
                tuple(float(value) for value in (turned.min(axis=0) + 0.5)),
                tuple(float(value) for value in (turned.max(axis=0) + 0.5)),
            ))
            element_faces = element.get("faces")
            if not isinstance(element_faces, dict):
                continue
            for slot in FACE_SLOTS:
                face = element_faces.get(slot)
                if not isinstance(face, dict):
                    continue
                normal, corners = element_face_corners(low, high, slot)
                world_normal = rotation @ (local_rotation @ normal)
                world_slot = _slot_for_normal(world_normal) or _nearest_slot_for_normal(world_normal)
                cull = face.get("cullface")
                cull_slot = None
                if isinstance(cull, str):
                    base = {"bottom": "down", "top": "up"}.get(cull, cull)
                    if base in FACE_SLOTS:
                        cull_slot = _slot_for_normal(
                            rotation @ np.asarray(FACE_BASIS[base][0], dtype=float)
                        )
                faces.append(
                    ElementFace(
                        world_slot,
                        ((corners - pivot) @ local_rotation.T + pivot) @ rotation.T,
                        world_normal,
                        self._element_face_texture(texture_map, face)
                        if include_textures
                        else None,
                        cull_slot,
                    )
                )
        if not rotated_corners:
            return None
        stacked = np.concatenate(rotated_corners, axis=0) + 0.5
        return ModelGeometry(
            tuple(faces),
            tuple(float(value) for value in stacked.min(axis=0)),
            tuple(float(value) for value in stacked.max(axis=0)),
            tuple(element_boxes),
        )


def compose(pieces: "list[ModelGeometry]") -> "ModelGeometry | None":
    """Merge several baked models into one shape.

    A multipart block is not one model but a set of them - a fence is a post plus
    a side for each neighbour it joins - so the pieces have to be unioned before
    the result means anything as a shape.
    """
    pieces = [piece for piece in pieces if piece is not None]
    if not pieces:
        return None
    faces = tuple(face for piece in pieces for face in piece.faces)
    boxes = tuple(box for piece in pieces for box in piece.boxes)
    if not boxes:
        return None
    minimum = tuple(min(box[0][axis] for box in boxes) for axis in range(3))
    maximum = tuple(max(box[1][axis] for box in boxes) for axis in range(3))
    return ModelGeometry(faces, minimum, maximum, boxes)


@lru_cache(maxsize=1)
def face_slot_order() -> tuple[str, ...]:
    return FACE_SLOTS


@dataclass(frozen=True, slots=True)
class ElementFace:
    """One outward face of one cuboid element of a block model."""

    slot: str
    """World face direction after the variant rotation."""
    corners: np.ndarray
    """``(4, 3)`` block-local corners, offsets from the block centre."""
    normal: np.ndarray
    """``(3,)`` outward unit normal in world axes."""
    texture: np.ndarray | None
    """``(16, 16, 3)`` float32 RGB for the face, or ``None`` when unresolved."""
    cullface: str | None
    """World direction whose neighbour hides this face, when the model says so."""


@dataclass(frozen=True, slots=True)
class ModelGeometry:
    """Every outward face of a block model, whatever shape the model is.

    ``bake`` deliberately handles only complete cubes, because the classifier
    compares full-cube appearances. Occlusion is a different question: a carpet
    one sixteenth of a block tall hides the whole face beneath it, and a model
    the atlas refuses to represent cannot hide anything at all. This keeps the
    shape as well as the appearance so partial blocks can be reasoned about.
    """

    faces: tuple[ElementFace, ...]
    minimum: tuple[float, float, float]
    """Lowest corner of the model's bounding box, in block units from 0 to 1."""
    maximum: tuple[float, float, float]
    """Highest corner of the model's bounding box, in block units from 0 to 1."""
    boxes: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...] = ()
    """Each element's own axis-aligned extent, after the variant rotation.

    Coverage has to be judged from these rather than from the bounding box: a
    staircase spans its whole cell corner to corner, so its box claims to hide
    every face, while the shape actually leaves half the top open.
    """

    def _filled(self) -> np.ndarray:
        """Which sixteenths of the cell the model occupies.

        Vanilla models are authored on a sixteenth-of-a-block grid, so sampling
        at that resolution is exact for them rather than an approximation.
        """
        grid = np.zeros((TEXTURE_SIZE, TEXTURE_SIZE, TEXTURE_SIZE), dtype=bool)
        for low, high in self.boxes:
            start = [int(np.floor(low[axis] * TEXTURE_SIZE + 1e-6)) for axis in range(3)]
            stop = [int(np.ceil(high[axis] * TEXTURE_SIZE - 1e-6)) for axis in range(3)]
            slices = tuple(
                slice(max(0, start[axis]), min(TEXTURE_SIZE, max(stop[axis], start[axis] + 1)))
                for axis in range(3)
            )
            grid[slices] = True
        return grid

    @property
    def full_cube(self) -> bool:
        """Whether the model fills its cell, and so hides everything behind it."""
        if not self.boxes:
            return False
        return bool(self._filled().all())

    @property
    def height(self) -> float:
        """How tall the model stands, which is what distinguishes a carpet from a slab."""
        return self.maximum[1] - self.minimum[1]

    def covers(self, slot: str) -> bool:
        """Whether the model completely covers its cell's face in the given direction.

        A slab covers the face it rests on but not the one above it; a staircase
        covers its back and its underside but only half its top. This is the test
        that decides whether the block behind can still be seen, so it has to
        follow the model's real shape.
        """
        if not self.boxes:
            return False
        axis = {"west": 0, "east": 0, "down": 1, "up": 1, "north": 2, "south": 2}[slot]
        layer = 0 if slot in ("west", "down", "north") else TEXTURE_SIZE - 1
        return bool(np.take(self._filled(), layer, axis=axis).all())


def element_face_corners(
    minimum: np.ndarray, maximum: np.ndarray, slot: str
) -> tuple[np.ndarray, np.ndarray]:
    """Outward normal and the four UV-ordered corners of one box face.

    Generalises ``FACE_BASIS``, which describes the unit cube, to an arbitrary
    box by substituting the box's own extent wherever the cube used 0 or 1. The
    corner order still starts at the one carrying texture pixel (0, 0), so a
    partial face's texture lines up the same way a full cube's does.
    """
    low, high = np.asarray(minimum, dtype=float), np.asarray(maximum, dtype=float)
    span = high - low
    layout: dict[str, tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]] = {
        # origin picker (0 means take the low corner, 1 the high), +u edge, +v edge
        "up": ((0, 1, 0), (span[0], 0, 0), (0, 0, span[2])),
        "down": ((0, 0, 1), (span[0], 0, 0), (0, 0, -span[2])),
        "north": ((1, 1, 0), (-span[0], 0, 0), (0, -span[1], 0)),
        "south": ((0, 1, 1), (span[0], 0, 0), (0, -span[1], 0)),
        "west": ((0, 1, 0), (0, 0, span[2]), (0, -span[1], 0)),
        "east": ((1, 1, 1), (0, 0, -span[2]), (0, -span[1], 0)),
    }
    picker, edge_u, edge_v = layout[slot]
    origin = np.asarray(
        [high[axis] if picker[axis] else low[axis] for axis in range(3)], dtype=float
    ) - 0.5
    edge_u = np.asarray(edge_u, dtype=float)
    edge_v = np.asarray(edge_v, dtype=float)
    corners = np.asarray([origin, origin + edge_u, origin + edge_u + edge_v, origin + edge_v])
    return np.asarray(FACE_BASIS[slot][0], dtype=float), corners


MULTIPART_LIMIT = 64
"""Most property combinations to enumerate for one multipart block.

A fence has four boolean sides, so sixteen; a wall has far more. Past this the
combinations describe joins a reconstruction cannot tell apart anyway.
"""


def _multipart_conditions(clause: object) -> list[dict[str, set[str]]]:
    """The alternatives one ``when`` clause accepts, as property to allowed values."""
    if not isinstance(clause, dict):
        return [{}]
    if "OR" in clause:
        alternatives: list[dict[str, set[str]]] = []
        for option in clause["OR"] if isinstance(clause["OR"], list) else []:
            alternatives.extend(_multipart_conditions(option))
        return alternatives or [{}]
    return [
        {
            str(name): {value.strip() for value in str(values).split("|")}
            for name, values in clause.items()
            if name != "OR"
        }
    ]


def blockstate_multipart(
    blockstate: object, limit: int = MULTIPART_LIMIT
) -> list[tuple[tuple[tuple[str, str], ...], tuple[tuple[str, float, float], ...]]]:
    """Yield ``(properties, parts)`` for a multipart blockstate.

    Fences, walls, panes and bars are not described as a list of variants but as a
    set of pieces with conditions - a post always, a side when that neighbour
    connects. `blockstate_variants` sees no ``variants`` key and returns nothing,
    which is why `dark_oak_fence` was absent from the palette entirely.

    The property domain is implied by the conditions rather than stated, so it is
    read back out of them and the combinations enumerated.
    """
    if not isinstance(blockstate, dict):
        return []
    entries = blockstate.get("multipart")
    if not isinstance(entries, list) or not entries:
        return []

    domain: dict[str, set[str]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for alternative in _multipart_conditions(entry.get("when")):
            for name, values in alternative.items():
                domain.setdefault(name, set()).update(values)
    # A condition only ever names the value that switches a piece on, so the off
    # value is implied by its absence: a fence mentions `north: true` and never
    # `north: false`, and reading the domain literally leaves one combination in
    # which every side connects.
    for name, values in domain.items():
        if values <= {"true", "false"}:
            values.update({"true", "false"})
        elif values & {"low", "tall"}:
            values.add("none")
    names = sorted(domain)
    combinations: list[tuple[tuple[str, str], ...]] = [()]
    for name in names:
        values = sorted(domain[name])
        widened = [
            existing + ((name, value),) for existing in combinations for value in values
        ]
        if len(widened) > limit:
            break
        combinations = widened

    out = []
    for properties in combinations:
        chosen = dict(properties)
        parts: list[tuple[str, float, float]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            apply = entry.get("apply")
            apply = apply[0] if isinstance(apply, list) and apply else apply
            if not isinstance(apply, dict) or not isinstance(apply.get("model"), str):
                continue
            when = entry.get("when")
            if when is not None:
                alternatives = _multipart_conditions(when)
                if not any(
                    all(chosen.get(name) in values for name, values in alternative.items())
                    for alternative in alternatives
                ):
                    continue
            parts.append(
                (apply["model"], float(apply.get("x", 0)), float(apply.get("y", 0)))
            )
        if parts:
            out.append((properties, tuple(parts)))
    return out
