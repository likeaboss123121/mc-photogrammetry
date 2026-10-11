"""Bounded, read-only native asset catalogue for a deliberately strict cube subset.

No appearance deduplication, resampling, placement or capture-pack certification.
One input archive must contain its own complete parent/texture dependencies.
Only minecraft namespace, deterministic exact-property variants, one opaque
static full cube, full canonical UVs and quarter-turn rotations are supported.
Unsupported states are atomic exclusions; a resource limit aborts the entire
catalogue rather than returning an apparently exhaustive prefix.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import json
import os
import re
import struct
import zipfile
import zlib

from PIL import Image

_SUBSET = 'minecraft-static-opaque-single-cube-canonical-uv-v1'


class CatalogueError(ValueError):
    """Archive/policy failure, including incomplete enumeration due to caps."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class _Unsupported(CatalogueError):
    pass


@dataclass(frozen=True, slots=True)
class CataloguePolicy:
    max_archive_bytes: int = 536_870_912
    max_members: int = 60_000
    max_total_uncompressed_bytes: int = 536_870_912
    max_member_bytes: int = 8_388_608
    max_json_bytes: int = 1_048_576
    max_texture_side: int = 128
    max_total_rgba_bytes: int = 67_108_864
    max_states: int = 8_192
    max_faces: int = 49_152
    max_parent_depth: int = 32
    max_work: int = 500_000

    def __post_init__(self):
        ceilings = (536_870_912, 60_000, 536_870_912, 8_388_608,
                    1_048_576, 128, 67_108_864, 8_192, 49_152, 32, 500_000)
        if any(type(v) is not int or not 0 < v <= c
               for v, c in zip(asdict(self).values(), ceilings)):
            raise CatalogueError('invalid_policy')


@dataclass(frozen=True, slots=True)
class AssetIdentity:
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class UnitFace:
    block_id: str
    properties: tuple[tuple[str, str], ...]
    variant_key: str
    model_id: str
    model_chain: tuple[AssetIdentity, ...]  # child to root
    blockstate_asset: AssetIdentity
    source_face: str
    face_slot: str  # rotated cardinal slot
    variant_rotation: tuple[int, int]  # vanilla x then y, negative angles
    uv: tuple[int, int, int, int]
    uv_rotation: int
    texture_id: str
    texture_asset: AssetIdentity  # original PNG bytes
    metadata_asset: AssetIdentity | None
    texture_size: tuple[int, int]  # height, width, original resolution
    rgba: bytes  # original decoded RGBA, no orientation/resizing applied
    rgba_sha256: str
    texture_corners: tuple[tuple[float, float, float], ...]
    # Centred cube vertices corresponding to raw texture TL, TR, BR, BL.
    sha256: str


@dataclass(frozen=True, slots=True)
class CatalogueExclusion:
    block_id: str
    variant_key: str | None
    reason: str
    asset_path: str


@dataclass(frozen=True, slots=True)
class UnitFaceCatalogue:
    faces: tuple[UnitFace, ...]
    exclusions: tuple[CatalogueExclusion, ...]
    source_sha256: str  # complete archive byte identity, not capture-pack proof
    policy: CataloguePolicy
    semantic_sha256: str  # decoded identities independent of ZIP ordering
    sha256: str  # semantic + exact source bytes + policy
    enumerated_states: int
    supported_states: int
    work_used: int
    complete_within_subset: bool = True
    supported_subset: str = _SUBSET


_BASIS = {
    'up': ((0, 1, 0), (-.5, .5, -.5), (1, 0, 0), (0, 0, 1)),
    'down': ((0, -1, 0), (-.5, -.5, .5), (1, 0, 0), (0, 0, -1)),
    'north': ((0, 0, -1), (.5, .5, -.5), (-1, 0, 0), (0, -1, 0)),
    'south': ((0, 0, 1), (-.5, .5, .5), (1, 0, 0), (0, -1, 0)),
    'west': ((-1, 0, 0), (-.5, .5, -.5), (0, 0, 1), (0, -1, 0)),
    'east': ((1, 0, 0), (.5, .5, .5), (0, 0, -1), (0, -1, 0)),
}
_SLOTS = tuple(sorted(_BASIS))
_TOKEN = re.compile(r'^[a-z0-9_]+$')
_RESOURCE = re.compile(r'^[a-z0-9_./-]+$')


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _digest(value):
    def encode(item):
        if isinstance(item, bytes):
            return {'sha256': _sha(item), 'bytes': len(item)}
        raise TypeError('unsupported_digest_value')
    return _sha(json.dumps(value, sort_keys=True, separators=(',', ':'),
                           allow_nan=False, default=encode).encode('utf-8'))


def _resource(value):
    if not isinstance(value, str) or len(value) > 256:
        raise _Unsupported('invalid_resource_id')
    namespace, path = value.split(':', 1) if ':' in value else ('minecraft', value)
    if namespace != 'minecraft':
        raise _Unsupported('unsupported_namespace')
    if not _RESOURCE.fullmatch(path) or any(p in ('', '.', '..') for p in path.split('/')):
        raise _Unsupported('invalid_resource_id')
    return 'minecraft:' + path


def _asset(resource, kind, suffix):
    return 'assets/minecraft/' + kind + '/' + resource.split(':', 1)[1] + suffix


def _quarter(value):
    if type(value) is not int or value not in (0, 90, 180, 270):
        raise _Unsupported('unsupported_rotation')
    return value


def _rotate(v, x, y):
    a, b, c = v
    for _ in range(x // 90):
        b, c = c, -b
    for _ in range(y // 90):
        a, c = -c, a
    return a, b, c


def _json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate_key')
        result[key] = value
    return result


class _Reader:
    def __init__(self, archive, policy):
        self.archive, self.policy = archive, policy
        self.work = self.states = self.rgba_bytes = 0
        self.json_cache, self.texture_cache = {}, {}
        self.members = {}
        infos = archive.infolist()
        self.tick(len(infos))
        if len(infos) > policy.max_members:
            raise CatalogueError('member_count_cap')
        total = 0
        for info in infos:
            name = info.filename
            if (len(name) > 512 or '\\' in name or name.startswith('/')
                    or any(p in ('.', '..') for p in name.split('/'))):
                raise CatalogueError('unsafe_member_path')
            if name in self.members:
                raise CatalogueError('duplicate_member')
            if info.flag_bits & 1:
                raise CatalogueError('encrypted_member')
            if info.file_size > policy.max_member_bytes:
                raise CatalogueError('member_bytes_cap')
            total += info.file_size
            if total > policy.max_total_uncompressed_bytes:
                raise CatalogueError('total_uncompressed_cap')
            self.members[name] = info
        if any('/optifine/' in n or '/mcpatcher/' in n or n.startswith('shaders/')
               for n in self.members):
            raise CatalogueError('unsupported_pack_extensions')

    def tick(self, count=1):
        self.work += count
        if self.work > self.policy.max_work:
            raise CatalogueError('work_cap')

    def read(self, path, json_asset=False):
        self.tick()
        if path not in self.members:
            raise _Unsupported('missing_asset')
        info = self.members[path]
        cap = self.policy.max_json_bytes if json_asset else self.policy.max_member_bytes
        if info.file_size > cap:
            raise CatalogueError('json_bytes_cap' if json_asset else 'member_bytes_cap')
        try:
            with self.archive.open(info) as stream:
                raw = stream.read(cap + 1)
        except (OSError, RuntimeError, NotImplementedError, zipfile.BadZipFile,
                EOFError, zlib.error) as exc:
            raise _Unsupported('corrupt_asset') from exc
        if len(raw) > cap or len(raw) != info.file_size:
            raise CatalogueError('member_bytes_cap')
        return raw, AssetIdentity(path, _sha(raw))

    def json(self, path):
        self.tick()
        if path not in self.json_cache:
            raw, identity = self.read(path, True)
            try:
                obj = json.loads(raw, object_pairs_hook=_json_pairs,
                                 parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
            except (ValueError, UnicodeError, RecursionError) as exc:
                raise _Unsupported('corrupt_json') from exc
            if not isinstance(obj, dict):
                raise _Unsupported('invalid_json_object')
            self.json_cache[path] = obj, identity
        return self.json_cache[path]

    def model(self, model_id):
        chain, seen, resolved = [], set(), {}
        current = model_id
        while True:
            self.tick()
            if current in seen:
                raise _Unsupported('model_parent_cycle')
            if len(chain) >= self.policy.max_parent_depth:
                raise CatalogueError('parent_depth_cap')
            seen.add(current)
            obj, identity = self.json(_asset(current, 'models', '.json'))
            if set(obj) - {'parent', 'textures', 'elements', 'ambientocclusion', 'display', 'gui_light'}:
                raise _Unsupported('unsupported_model_semantics')
            chain.append((obj, identity))
            if 'parent' not in obj:
                break
            current = _resource(obj['parent'])
        texture_map = {}
        for obj, _ in reversed(chain):
            if 'textures' in obj:
                if not isinstance(obj['textures'], dict):
                    raise _Unsupported('invalid_texture_map')
                texture_map.update(obj['textures'])
            if 'elements' in obj:
                resolved['elements'] = obj['elements']
        return resolved.get('elements'), texture_map, tuple(i for _, i in chain)

    def texture(self, texture_id):
        self.tick()
        if texture_id in self.texture_cache:
            return self.texture_cache[texture_id]
        path = _asset(texture_id, 'textures', '.png')
        metadata = None
        if path + '.mcmeta' in self.members:
            obj, metadata = self.json(path + '.mcmeta')
            if 'animation' in obj:
                raise _Unsupported('animated_texture')
            if obj:
                raise _Unsupported('unsupported_texture_metadata')
        raw, identity = self.read(path)
        try:
            with Image.open(io.BytesIO(raw)) as image:
                if image.format != 'PNG':
                    raise _Unsupported('non_png_texture')
                w, h = image.size
                if w <= 0 or h <= 0 or max(w, h) > self.policy.max_texture_side:
                    raise CatalogueError('texture_side_cap')
                if getattr(image, 'n_frames', 1) != 1:
                    raise _Unsupported('animated_texture')
                if w != h:
                    raise _Unsupported('non_square_texture')
                cost = w * h * 4
                if self.rgba_bytes + cost > self.policy.max_total_rgba_bytes:
                    raise CatalogueError('rgba_bytes_cap')
                image.load()
                rgba = image.convert('RGBA').tobytes()
        except Image.DecompressionBombError as exc:
            raise CatalogueError('texture_side_cap') from exc
        except (OSError, SyntaxError, ValueError, zlib.error) as exc:
            if isinstance(exc, CatalogueError):
                raise
            raise _Unsupported('corrupt_texture') from exc
        if any(alpha != 255 for alpha in rgba[3::4]):
            raise _Unsupported('non_opaque_texture')
        self.rgba_bytes += cost
        result = identity, metadata, (h, w), rgba
        self.texture_cache[texture_id] = result
        return result


def _properties(key):
    if not isinstance(key, str) or len(key) > 512:
        raise _Unsupported('conditional_variant')
    if key == '':
        return ()
    props = []
    for part in key.split(','):
        if part.count('=') != 1:
            raise _Unsupported('conditional_variant')
        name, value = part.split('=')
        if not _TOKEN.fullmatch(name) or not _TOKEN.fullmatch(value):
            raise _Unsupported('conditional_variant')
        props.append((name, value))
    if len({name for name, _ in props}) != len(props):
        raise _Unsupported('conditional_variant')
    return tuple(sorted(props))


def _faces(reader, block_id, key, props, variant, blockstate):
    if isinstance(variant, list):
        raise _Unsupported('random_variant')
    if not isinstance(variant, dict):
        raise _Unsupported('invalid_variant')
    if 'weight' in variant:
        raise _Unsupported('weighted_variant')
    if set(variant) - {'model', 'x', 'y', 'uvlock'}:
        raise _Unsupported('unsupported_variant_semantics')
    if 'uvlock' in variant and type(variant['uvlock']) is not bool:
        raise _Unsupported('invalid_uvlock')
    if variant.get('uvlock', False):
        raise _Unsupported('uvlock_variant')
    model_id = _resource(variant.get('model'))
    x, y = _quarter(variant.get('x', 0)), _quarter(variant.get('y', 0))
    elements, textures, chain = reader.model(model_id)
    if not isinstance(elements, list) or len(elements) != 1:
        raise _Unsupported('partial_model')
    element = elements[0]
    if not isinstance(element, dict) or set(element) - {'from', 'to', 'faces', 'shade'}:
        raise _Unsupported('unsupported_element_semantics')
    if (element.get('from') != [0, 0, 0] or element.get('to') != [16, 16, 16]
            or any(type(n) not in (int, float) for n in element.get('from', [])
                   + element.get('to', []))):
        raise _Unsupported('partial_model')
    if 'shade' in element and type(element['shade']) is not bool:
        raise _Unsupported('unsupported_element_semantics')
    faces = element.get('faces')
    if not isinstance(faces, dict) or set(faces) != set(_SLOTS):
        raise _Unsupported('incomplete_cube_faces')
    result = []
    for slot in _SLOTS:
        reader.tick()
        face = faces[slot]
        if not isinstance(face, dict) or set(face) - {'texture', 'uv', 'rotation', 'cullface', 'tintindex'}:
            raise _Unsupported('unsupported_face_semantics')
        if 'tintindex' in face and (type(face['tintindex']) is not int or face['tintindex'] != -1):
            raise _Unsupported('tinted_face')
        if 'cullface' in face and face['cullface'] != slot:
            raise _Unsupported('unsupported_cullface')
        uv = face.get('uv', [0, 0, 16, 16])
        if (not isinstance(uv, list) or uv != [0, 0, 16, 16]
                or any(type(n) not in (int, float) for n in uv)):
            raise _Unsupported('noncanonical_uv')
        turn = _quarter(face.get('rotation', 0))
        texture, seen = face.get('texture'), set()
        while isinstance(texture, str) and texture.startswith('#'):
            reader.tick()
            if texture in seen:
                raise _Unsupported('texture_reference_cycle')
            seen.add(texture)
            if texture[1:] not in textures:
                raise _Unsupported('missing_texture_reference')
            texture = textures[texture[1:]]
        texture_id = _resource(texture)
        texture_asset, metadata, size, rgba = reader.texture(texture_id)
        normal, origin, du, dv = _BASIS[slot]
        corners = (origin, tuple(a+b for a, b in zip(origin, du)),
                   tuple(a+b+c for a, b, c in zip(origin, du, dv)),
                   tuple(a+b for a, b in zip(origin, dv)))
        # Vanilla clockwise UV rotation moves raw TL to geometric TR at +90.
        # Return geometry for raw TL/TR/BR/BL, then apply x followed by y.
        corners = tuple(_rotate(corners[(i + turn // 90) % 4], x, y) for i in range(4))
        world_normal = _rotate(normal, x, y)
        world_slot = next(s for s in _SLOTS if _BASIS[s][0] == world_normal)
        fields = dict(block_id=block_id, properties=props, variant_key=key,
                      model_id=model_id, model_chain=chain, blockstate_asset=blockstate,
                      source_face=slot, face_slot=world_slot, variant_rotation=(x, y),
                      uv=(0, 0, 16, 16), uv_rotation=turn, texture_id=texture_id,
                      texture_asset=texture_asset, metadata_asset=metadata,
                      texture_size=size, rgba=rgba, rgba_sha256=_sha(rgba),
                      texture_corners=corners)
        digest_fields = dict(fields)
        digest_fields['model_chain'] = [asdict(a) for a in chain]
        digest_fields['blockstate_asset'] = asdict(blockstate)
        digest_fields['texture_asset'] = asdict(texture_asset)
        digest_fields['metadata_asset'] = asdict(metadata) if metadata else None
        result.append(UnitFace(**fields, sha256=_digest(digest_fields)))
    return result


def _archive_bytes(source, policy):
    if isinstance(source, bytes):
        raw = source
    elif isinstance(source, (str, os.PathLike)):
        try:
            with open(source, 'rb') as stream:
                raw = stream.read(policy.max_archive_bytes + 1)
        except OSError as exc:
            raise CatalogueError('unreadable_source') from exc
    else:
        raise CatalogueError('invalid_source')
    if len(raw) > policy.max_archive_bytes:
        raise CatalogueError('archive_bytes_cap')
    # Bound central-directory allocation before ZipFile builds its object list.
    tail_start = max(0, len(raw) - 65_557)
    eocd = raw.rfind(b'PK\x05\x06', tail_start)
    if eocd < 0 or len(raw) - eocd < 22:
        raise CatalogueError('corrupt_archive')
    disk, cd_disk, disk_n, n, cd_size, cd_offset, comment_n = struct.unpack_from('<4H2IH', raw, eocd+4)
    if disk or cd_disk or disk_n != n:
        raise CatalogueError('unsupported_multidisk_archive')
    if n == 65_535 or cd_size == 0xffffffff or cd_offset == 0xffffffff:
        raise CatalogueError('unsupported_zip64_archive')
    if eocd + 22 + comment_n != len(raw) or cd_offset + cd_size != eocd:
        raise CatalogueError('corrupt_archive')
    if n > policy.max_members:
        raise CatalogueError('member_count_cap')
    # A forged count cannot evade the pre-allocation cap.
    cursor, actual = cd_offset, 0
    while cursor < eocd:
        if raw[cursor:cursor+4] != b'PK\x01\x02' or cursor+46 > eocd:
            raise CatalogueError('corrupt_archive')
        name_n, extra_n, note_n = struct.unpack_from('<3H', raw, cursor+28)
        name_raw = raw[cursor+46:cursor+46+name_n]
        # ZipInfo normalizes backslashes on Windows and truncates NULs; inspect
        # the original central-directory spelling before that lossy conversion.
        if b'\\' in name_raw or b'\x00' in name_raw:
            raise CatalogueError('unsafe_member_path')
        cursor += 46 + name_n + extra_n + note_n
        actual += 1
        if actual > policy.max_members:
            raise CatalogueError('member_count_cap')
    if cursor != eocd or actual != n:
        raise CatalogueError('corrupt_archive')
    return raw


def load_unit_face_catalogue(asset_source, fixed_policy: CataloguePolicy | None = None):
    """Read local path or ZIP bytes, returning immutable records and exclusions.

    Fatal archive/resource errors raise CatalogueError. Unsupported or corrupt
    individual states are retained as exclusions. No fallback pack, cache write,
    screenshot selection or external dependency lookup is performed.
    """
    policy = fixed_policy if fixed_policy is not None else CataloguePolicy()
    if type(policy) is not CataloguePolicy:
        raise CatalogueError('invalid_policy')
    raw = _archive_bytes(asset_source, policy)
    faces, exclusions = [], []
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            reader = _Reader(archive, policy)
            blockstates = sorted(n for n in reader.members
                                 if n.startswith('assets/minecraft/blockstates/') and n.endswith('.json'))
            if not blockstates:
                raise CatalogueError('no_blockstates')
            for path in blockstates:
                name = path[len('assets/minecraft/blockstates/'):-5]
                block_id = 'minecraft:' + name
                try:
                    _resource(block_id)
                    obj, identity = reader.json(path)
                    if 'multipart' in obj:
                        raise _Unsupported('multipart_blockstate')
                    if set(obj) != {'variants'} or not isinstance(obj['variants'], dict) or not obj['variants']:
                        raise _Unsupported('unsupported_blockstate')
                    variants = obj['variants']
                    keys = sorted(variants)
                    props = [_properties(key) for key in keys]
                    if (len(set(props)) != len(props)
                            or len({tuple(n for n, _ in p) for p in props}) != 1):
                        raise _Unsupported('conditional_variants')
                except _Unsupported as exc:
                    exclusions.append(CatalogueExclusion(block_id, None, exc.reason, path))
                    continue
                for key, properties in zip(keys, props):
                    reader.tick()
                    reader.states += 1
                    if reader.states > policy.max_states:
                        raise CatalogueError('state_count_cap')
                    try:
                        state_faces = _faces(reader, block_id, key, properties, variants[key], identity)
                    except _Unsupported as exc:
                        exclusions.append(CatalogueExclusion(block_id, key, exc.reason, path))
                        continue
                    if len(faces) + len(state_faces) > policy.max_faces:
                        raise CatalogueError('face_count_cap')
                    faces.extend(state_faces)
    except (zipfile.BadZipFile, OSError, EOFError, UnicodeError) as exc:
        raise CatalogueError('corrupt_archive') from exc
    # Enumeration order is canonical block path, raw variant key, source face.
    faces_tuple, excluded_tuple = tuple(faces), tuple(exclusions)
    semantic = _digest({'subset': _SUBSET,
                        'faces': [f.sha256 for f in faces_tuple],
                        'exclusions': [asdict(e) for e in excluded_tuple]})
    source = _sha(raw)
    digest = _digest({'semantic': semantic, 'source': source, 'policy': asdict(policy)})
    return UnitFaceCatalogue(faces_tuple, excluded_tuple, source, policy, semantic,
                             digest, reader.states, len(faces) // 6, reader.work)
