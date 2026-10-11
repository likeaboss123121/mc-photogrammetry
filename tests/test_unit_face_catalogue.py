"""Generated assets only; no project JAR, screenshots or evaluator payloads."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import hashlib
import io
import json
import struct
import zipfile
import zlib

from PIL import Image, PngImagePlugin
import pytest

from mc_photogrammetry.blocks.unit_face_catalogue import (
    CatalogueError, CataloguePolicy, load_unit_face_catalogue,
)


SLOTS = ('down', 'up', 'north', 'south', 'west', 'east')
MODEL = 'assets/minecraft/models/block/cube.json'
STATE = 'assets/minecraft/blockstates/test.json'
TEXTURE = 'assets/minecraft/textures/block/pattern.png'


def png(side=16, alpha=255):
    image = Image.new('RGBA', (side, side))
    image.putdata([(x * 7 % 256, y * 11 % 256, (x+y) * 13 % 256, alpha)
                   for y in range(side) for x in range(side)])
    stream = io.BytesIO()
    image.save(stream, format='PNG')
    return stream.getvalue()


def cube():
    return {'textures': {'all': 'minecraft:block/pattern'}, 'elements': [{
        'from': [0, 0, 0], 'to': [16, 16, 16],
        'faces': {s: {'texture': '#all', 'cullface': s} for s in SLOTS},
    }]}


def assets():
    return {MODEL: cube(), STATE: {'variants': {'': {'model': 'block/cube'}}},
            TEXTURE: png()}


def archive(items, reverse=False):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as zipped:
        for key in sorted(items, reverse=reverse):
            value = items[key]
            if not isinstance(value, (str, bytes)):
                value = json.dumps(value, separators=(',', ':'), sort_keys=True)
            zipped.writestr(key, value)
    return stream.getvalue()


def load(items=None, **caps):
    return load_unit_face_catalogue(archive(assets() if items is None else items),
                                    replace(CataloguePolicy(), **caps))


def excluded(items, reason):
    result = load(items)
    assert not result.faces
    assert result.supported_states == 0
    assert len(result.exclusions) == 1
    assert result.exclusions[0].reason == reason
    assert result.complete_within_subset
    return result


def test_full_cube_implicit_uv_native_bytes_provenance_and_immutability(tmp_path):
    items = assets()
    raw = archive(items)
    path = tmp_path / 'pack.zip'
    path.write_bytes(raw)
    result = load_unit_face_catalogue(path)
    assert result == load_unit_face_catalogue(raw)
    assert result.enumerated_states == result.supported_states == 1
    assert len(result.faces) == 6
    assert not result.exclusions
    assert [f.source_face for f in result.faces] == sorted(SLOTS)
    with Image.open(io.BytesIO(items[TEXTURE])) as image:
        expected = image.convert('RGBA').tobytes()
    for face in result.faces:
        assert face.rgba == expected
        assert face.texture_size == (16, 16)
        assert face.uv == (0, 0, 16, 16)
        assert face.block_id == 'minecraft:test'
        assert face.model_chain[0].path == MODEL
        assert face.blockstate_asset.path == STATE
        assert face.texture_asset.path == TEXTURE
        assert len(face.sha256) == len(face.rgba_sha256) == 64
        assert len(set(face.texture_corners)) == 4
        assert all(all(n in (-.5, .5) for n in c) for c in face.texture_corners)
    with pytest.raises(FrozenInstanceError):
        result.faces = ()
    with pytest.raises(FrozenInstanceError):
        result.faces[0].rgba = b''
    assert list(tmp_path.iterdir()) == [path]  # no cache writes


def test_inherited_elements_and_child_texture_aliases():
    items = assets()
    items[MODEL]['textures'] = {'all': '#surface', 'surface': 'block/pattern'}
    child = 'assets/minecraft/models/block/child.json'
    items[child] = {'parent': 'block/cube', 'textures': {'surface': 'block/other'}}
    other = 'assets/minecraft/textures/block/other.png'
    items[other] = png(32)
    items[STATE]['variants']['']['model'] = 'block/child'
    result = load(items)
    assert len(result.faces) == 6
    for face in result.faces:
        assert tuple(a.path for a in face.model_chain) == (child, MODEL)
        assert face.texture_asset.path == other
        assert face.texture_id == 'minecraft:block/other'
        assert face.texture_size == (32, 32)
        assert len(face.rgba) == 32*32*4


def test_child_elements_override_parent_without_merging():
    items = assets()
    items['assets/minecraft/models/block/child.json'] = {
        'parent': 'block/cube', 'elements': [],
    }
    items[STATE]['variants']['']['model'] = 'block/child'
    excluded(items, 'partial_model')


@pytest.mark.parametrize(('x', 'y', 'world'), [(0, 90, 'east'), (90, 0, 'down'),
                                             (180, 0, 'south'), (0, 270, 'west')])
def test_variant_rotation_face_slot_and_raw_texture_identity(x, y, world):
    base = load()
    items = assets()
    items[STATE]['variants'][''].update(x=x, y=y)
    result = load(items)
    face = next(f for f in result.faces if f.source_face == 'north')
    assert face.face_slot == world
    assert face.variant_rotation == (x, y)
    assert face.rgba == base.faces[0].rgba  # never rotate original RGBA
    axis, sign = {'east': (0, .5), 'west': (0, -.5),
                  'down': (1, -.5), 'south': (2, .5)}[world]
    assert all(c[axis] == sign for c in face.texture_corners)
    assert {f.face_slot for f in result.faces} == set(SLOTS)


# Eight explicitly named cube vertices: west/east, down/up, north/south.
WDN, WDS = (-.5, -.5, -.5), (-.5, -.5, .5)
WUN, WUS = (-.5, .5, -.5), (-.5, .5, .5)
EDN, EDS = (.5, -.5, -.5), (.5, -.5, .5)
EUN, EUS = (.5, .5, -.5), (.5, .5, .5)

# Each literal row gives the geometry at raw TL, TR, BR, BL for UV turns
# 0, 90, 180, 270 respectively. Positive UV turns move raw TL clockwise
# to geometric TR. The combined block rotations are x first, then y:
# (90, 90) sends the positive x/y/z axes to +z/+x/+y;
# (270, 90) sends them to +z/-x/-y. No implementation basis or transform
# helper participates in these independently enumerated expectations.
RAW_CORNER_EXPECTATIONS = {
    (0, 0): {
        'down': ('down', (
            (WDS, EDS, EDN, WDN), (EDS, EDN, WDN, WDS),
            (EDN, WDN, WDS, EDS), (WDN, WDS, EDS, EDN))),
        'up': ('up', (
            (WUN, EUN, EUS, WUS), (EUN, EUS, WUS, WUN),
            (EUS, WUS, WUN, EUN), (WUS, WUN, EUN, EUS))),
        'north': ('north', (
            (EUN, WUN, WDN, EDN), (WUN, WDN, EDN, EUN),
            (WDN, EDN, EUN, WUN), (EDN, EUN, WUN, WDN))),
        'south': ('south', (
            (WUS, EUS, EDS, WDS), (EUS, EDS, WDS, WUS),
            (EDS, WDS, WUS, EUS), (WDS, WUS, EUS, EDS))),
        'west': ('west', (
            (WUN, WUS, WDS, WDN), (WUS, WDS, WDN, WUN),
            (WDS, WDN, WUN, WUS), (WDN, WUN, WUS, WDS))),
        'east': ('east', (
            (EUS, EUN, EDN, EDS), (EUN, EDN, EDS, EUS),
            (EDN, EDS, EUS, EUN), (EDS, EUS, EUN, EDN))),
    },
    (90, 90): {
        'down': ('west', (
            (WUN, WUS, WDS, WDN), (WUS, WDS, WDN, WUN),
            (WDS, WDN, WUN, WUS), (WDN, WUN, WUS, WDS))),
        'up': ('east', (
            (EDN, EDS, EUS, EUN), (EDS, EUS, EUN, EDN),
            (EUS, EUN, EDN, EDS), (EUN, EDN, EDS, EUS))),
        'north': ('down', (
            (EDS, EDN, WDN, WDS), (EDN, WDN, WDS, EDS),
            (WDN, WDS, EDS, EDN), (WDS, EDS, EDN, WDN))),
        'south': ('up', (
            (EUN, EUS, WUS, WUN), (EUS, WUS, WUN, EUN),
            (WUS, WUN, EUN, EUS), (WUN, EUN, EUS, WUS))),
        'west': ('north', (
            (EDN, EUN, WUN, WDN), (EUN, WUN, WDN, EDN),
            (WUN, WDN, EDN, EUN), (WDN, EDN, EUN, WUN))),
        'east': ('south', (
            (EUS, EDS, WDS, WUS), (EDS, WDS, WUS, EUS),
            (WDS, WUS, EUS, EDS), (WUS, EUS, EDS, WDS))),
    },
    (270, 90): {
        'down': ('east', (
            (EDN, EDS, EUS, EUN), (EDS, EUS, EUN, EDN),
            (EUS, EUN, EDN, EDS), (EUN, EDN, EDS, EUS))),
        'up': ('west', (
            (WUN, WUS, WDS, WDN), (WUS, WDS, WDN, WUN),
            (WDS, WDN, WUN, WUS), (WDN, WUN, WUS, WDS))),
        'north': ('up', (
            (WUS, WUN, EUN, EUS), (WUN, EUN, EUS, WUS),
            (EUN, EUS, WUS, WUN), (EUS, WUS, WUN, EUN))),
        'south': ('down', (
            (WDN, WDS, EDS, EDN), (WDS, EDS, EDN, WDN),
            (EDS, EDN, WDN, WDS), (EDN, WDN, WDS, EDS))),
        'west': ('north', (
            (WUN, WDN, EDN, EUN), (WDN, EDN, EUN, WUN),
            (EDN, EUN, WUN, WDN), (EUN, WUN, WDN, EDN))),
        'east': ('south', (
            (WDS, WUS, EUS, EDS), (WUS, EUS, EDS, WDS),
            (EUS, EDS, WDS, WUS), (EDS, WDS, WUS, EUS))),
    },
}


@pytest.mark.parametrize(('rotation', 'source', 'turn', 'world', 'expected'), [
    (rotation, source, turn, world, corners)
    for rotation, faces in RAW_CORNER_EXPECTATIONS.items()
    for source, (world, rows) in faces.items()
    for turn, corners in zip((0, 90, 180, 270), rows)
])
def test_uv_rotation_binds_independent_asymmetric_raw_corners(
        rotation, source, turn, world, expected):
    items = assets()
    # Distinct colors identify raw TL=red, TR=green, BR=blue, BL=yellow.
    # RGBA storage is row-major TL, TR, BL, BR and must remain unrotated.
    raw_rgba = bytes((255, 0, 0, 255, 0, 255, 0, 255,
                      255, 255, 0, 255, 0, 0, 255, 255))
    stream = io.BytesIO()
    Image.frombytes('RGBA', (2, 2), raw_rgba).save(stream, format='PNG')
    items[TEXTURE] = stream.getvalue()
    baseline = next(f for f in load(items).faces if f.source_face == source)
    x, y = rotation
    items[STATE]['variants'][''].update(x=x, y=y)
    items[MODEL]['elements'][0]['faces'][source]['rotation'] = turn
    face = next(f for f in load(items).faces if f.source_face == source)
    assert face.texture_corners == expected
    assert face.face_slot == world
    assert face.variant_rotation == rotation
    assert face.uv_rotation == turn
    assert face.texture_size == (2, 2)
    assert face.rgba == baseline.rgba == raw_rgba
    assert face.rgba_sha256 == baseline.rgba_sha256 == hashlib.sha256(raw_rgba).hexdigest()
    assert face.texture_asset == baseline.texture_asset
    assert face.texture_asset.sha256 == hashlib.sha256(items[TEXTURE]).hexdigest()
    assert face.texture_id == baseline.texture_id == 'minecraft:block/pattern'
    if rotation != (0, 0) or turn != 0:
        assert face.sha256 != baseline.sha256


def test_all_property_states_and_same_rgba_distinct_identities_retained():
    items = assets()
    items[STATE]['variants'] = {
        'axis=x': {'model': 'block/cube', 'x': 90},
        'axis=y': {'model': 'block/cube'},
        'axis=z': {'model': 'block/cube', 'y': 90},
    }
    result = load(items)
    assert result.supported_states == result.enumerated_states == 3
    assert len(result.faces) == len({f.sha256 for f in result.faces}) == 18
    assert len({f.rgba_sha256 for f in result.faces}) == 1
    assert {f.properties for f in result.faces} == {
        (('axis', 'x'),), (('axis', 'y'),), (('axis', 'z'),),
    }


def test_different_models_textures_and_property_order_are_preserved():
    items = assets()
    items['assets/minecraft/models/block/other.json'] = deepcopy(items[MODEL])
    items['assets/minecraft/models/block/other.json']['textures']['all'] = 'block/other'
    items['assets/minecraft/textures/block/other.png'] = png(32)
    items[STATE]['variants'] = {
        'lit=false,axis=x': {'model': 'block/cube'},
        'axis=x,lit=true': {'model': 'block/other'},
    }
    result = load(items)
    assert len(result.faces) == 12
    assert {f.texture_size for f in result.faces} == {(16, 16), (32, 32)}
    assert {f.model_id for f in result.faces} == {'minecraft:block/cube', 'minecraft:block/other'}


@pytest.mark.parametrize(('variant', 'reason'), [
    ([{'model': 'block/cube'}], 'random_variant'),
    ([{'model': 'block/cube', 'weight': 1}, {'model': 'block/cube'}], 'random_variant'),
    ({'model': 'block/cube', 'weight': 1}, 'weighted_variant'),
    ({'model': 'block/cube', 'uvlock': True}, 'uvlock_variant'),
    ({'model': 'block/cube', 'uvlock': 1}, 'invalid_uvlock'),
    ({'model': 'block/cube', 'x': 45}, 'unsupported_rotation'),
    ({'model': 'block/cube', 'y': 90.0}, 'unsupported_rotation'),
    ({'model': 'other:block/cube'}, 'unsupported_namespace'),
    ({'model': '../cube'}, 'invalid_resource_id'),
    ({'model': 'block/cube', 'condition': True}, 'unsupported_variant_semantics'),
])
def test_unsupported_variants_are_atomic_exclusions(variant, reason):
    items = assets()
    items[STATE]['variants'][''] = variant
    excluded(items, reason)


@pytest.mark.parametrize('key', ['axis=x|y', 'axis!=x', 'axis=x,axis=y', 'north',
                               'axis= x', 'axis=x,'])
def test_conditional_keys_excluded(key):
    items = assets()
    items[STATE]['variants'] = {key: {'model': 'block/cube'}}
    excluded(items, 'conditional_variant')


@pytest.mark.parametrize('keys', [('', 'axis=x'), ('axis=x,lit=true', 'lit=true,axis=x')])
def test_overlapping_or_duplicate_state_selectors_excluded(keys):
    items = assets()
    items[STATE]['variants'] = {k: {'model': 'block/cube'} for k in keys}
    excluded(items, 'conditional_variants')


def test_multipart_and_extensions_refused():
    items = assets()
    items[STATE] = {'multipart': [{'apply': {'model': 'block/cube'}}]}
    excluded(items, 'multipart_blockstate')
    items = assets()
    items['assets/minecraft/optifine/ctm/test.properties'] = 'method=ctm'
    with pytest.raises(CatalogueError, match='unsupported_pack_extensions'):
        load(items)


@pytest.mark.parametrize('uv', [[0, 0, 8, 16], [16, 0, 0, 16], [0, 0, 16, 8],
                             [0, 16, 16, 0], [False, 0, 16, 16], None])
def test_cropped_reversed_malformed_uv_refused(uv):
    items = assets()
    items[MODEL]['elements'][0]['faces']['north']['uv'] = uv
    excluded(items, 'noncanonical_uv')


def test_explicit_full_uv_supported_and_identity_binds_model_bytes():
    result = load()
    items = assets()
    for face in items[MODEL]['elements'][0]['faces'].values():
        face['uv'] = [0, 0, 16, 16]
    explicit = load(items)
    assert len(explicit.faces) == 6
    assert explicit.semantic_sha256 != result.semantic_sha256
    assert [f.texture_corners for f in explicit.faces] == [f.texture_corners for f in result.faces]


@pytest.mark.parametrize(('edit', 'reason'), [
    ('partial', 'partial_model'), ('multiple', 'partial_model'),
    ('missingface', 'incomplete_cube_faces'), ('rotation', 'unsupported_element_semantics'),
    ('tint', 'tinted_face'), ('cull', 'unsupported_cullface'),
    ('loader', 'unsupported_model_semantics'),
])
def test_unsupported_geometry_or_dynamic_appearance(edit, reason):
    items = assets()
    model = items[MODEL]
    element = model['elements'][0]
    if edit == 'partial':
        element['to'] = [16, 8, 16]
    elif edit == 'multiple':
        model['elements'].append(deepcopy(element))
    elif edit == 'missingface':
        del element['faces']['north']
    elif edit == 'rotation':
        element['rotation'] = {'axis': 'y', 'angle': 0}
    elif edit == 'tint':
        element['faces']['north']['tintindex'] = 0
    elif edit == 'cull':
        element['faces']['north']['cullface'] = 'south'
    else:
        model['loader'] = 'forge:custom'
    excluded(items, reason)


@pytest.mark.parametrize('alpha', [0, 128, 254])
def test_native_partial_alpha_refused(alpha):
    items = assets()
    items[TEXTURE] = png(alpha=alpha)
    excluded(items, 'non_opaque_texture')


@pytest.mark.parametrize('metadata', [{'animation': {}}, {'animation': {'frames': [0]}},
                                   {'texture': {'blur': True}}])
def test_animation_and_nontrivial_metadata_refused(metadata):
    items = assets()
    items[TEXTURE+'.mcmeta'] = metadata
    excluded(items, 'animated_texture' if 'animation' in metadata else 'unsupported_texture_metadata')


def test_empty_metadata_bound_and_non_square_strip_refused():
    items = assets()
    items[TEXTURE+'.mcmeta'] = {}
    result = load(items)
    assert result.faces[0].metadata_asset.path == TEXTURE+'.mcmeta'
    stream = io.BytesIO()
    Image.new('RGBA', (16, 32), (0, 0, 0, 255)).save(stream, format='PNG')
    items[TEXTURE] = stream.getvalue()
    excluded(items, 'non_square_texture')


@pytest.mark.parametrize(('edit', 'reason'), [
    ('parentcycle', 'model_parent_cycle'), ('texturecycle', 'texture_reference_cycle'),
    ('missingparent', 'missing_asset'), ('missingtexture', 'missing_asset'),
    ('missingref', 'missing_texture_reference'), ('badjson', 'corrupt_json'),
    ('duplicatekey', 'corrupt_json'), ('nan', 'corrupt_json'),
    ('badpng', 'corrupt_texture'),
])
def test_missing_cyclic_corrupt_assets(edit, reason):
    items = assets()
    if edit == 'parentcycle':
        items[MODEL]['parent'] = 'block/child'
        items['assets/minecraft/models/block/child.json'] = {'parent': 'block/cube'}
    elif edit == 'texturecycle':
        items[MODEL]['textures'] = {'all': '#a', 'a': '#all'}
    elif edit == 'missingparent':
        items[MODEL]['parent'] = 'block/missing'
    elif edit == 'missingtexture':
        del items[TEXTURE]
    elif edit == 'missingref':
        items[MODEL]['textures'] = {}
    elif edit == 'badjson':
        items[MODEL] = b'{'
    elif edit == 'duplicatekey':
        items[MODEL] = b'{"elements":[],"elements":[]}'
    elif edit == 'nan':
        items[MODEL] = b'{"elements":NaN}'
    else:
        items[TEXTURE] = b'not a png'
    excluded(items, reason)


@pytest.mark.parametrize('decoder_error', ['native', 'zlib'])
def test_valid_png_header_bad_idat_is_typed_atomic_exclusion(monkeypatch, decoder_error):
    def chunk(tag, data):
        return (struct.pack('>I', len(data)) + tag + data
                + struct.pack('>I', zlib.crc32(tag + data)))

    # Complete PNG signature, valid IHDR/IEND and valid chunk CRCs isolate
    # corrupt compressed image data from malformed container/header errors.
    bad_idat = b'not-zlib'
    raw = (b'\x89PNG\r\n\x1a\n'
           + chunk(b'IHDR', struct.pack('>2I5B', 2, 2, 8, 6, 0, 0, 0))
           + chunk(b'IDAT', bad_idat) + chunk(b'IEND', b''))
    with pytest.raises(zlib.error):
        zlib.decompress(bad_idat)
    with Image.open(io.BytesIO(raw)) as image:
        assert image.format == 'PNG'
        assert image.size == (2, 2)
        assert image.n_frames == 1
    if decoder_error == 'zlib':
        # Pillow's native decoder here wraps corruption as OSError; cover
        # a decoder exposing the underlying zlib.error with the same PNG.
        native_load = PngImagePlugin.PngImageFile.load

        def load_with_zlib_error(image, *args, **kwargs):
            try:
                return native_load(image, *args, **kwargs)
            except OSError as exc:
                raise zlib.error('invalid compressed IDAT') from exc

        monkeypatch.setattr(PngImagePlugin.PngImageFile, 'load', load_with_zlib_error)
    items = assets()
    items[TEXTURE] = raw
    result = excluded(items, 'corrupt_texture')
    assert result.enumerated_states == 1
    assert result.exclusions[0].asset_path == STATE


def test_unsupported_one_state_does_not_discard_other_supported_states():
    items = assets()
    items[STATE]['variants'] = {'axis=x': {'model': 'block/cube'},
                               'axis=y': [{'model': 'block/cube'}]}
    result = load(items)
    assert result.enumerated_states == 2
    assert result.supported_states == 1
    assert len(result.faces) == 6
    assert result.exclusions[0].variant_key == 'axis=y'


def test_deterministic_digest_and_order_independent_semantics():
    items = assets()
    raw = archive(items)
    a, b = load_unit_face_catalogue(raw), load_unit_face_catalogue(raw)
    assert a == b
    reordered = load_unit_face_catalogue(archive(items, reverse=True))
    assert a.faces == reordered.faces
    assert a.semantic_sha256 == reordered.semantic_sha256
    assert a.source_sha256 != reordered.source_sha256
    assert a.sha256 != reordered.sha256  # exact container identity is retained
    policy_change = load_unit_face_catalogue(raw, replace(CataloguePolicy(), max_work=400_000))
    assert a.semantic_sha256 == policy_change.semantic_sha256
    assert a.sha256 != policy_change.sha256
    items[TEXTURE] = png(32)
    assert load(items).semantic_sha256 != a.semantic_sha256


@pytest.mark.parametrize(('caps', 'reason'), [
    ({'max_archive_bytes': 20}, 'archive_bytes_cap'),
    ({'max_members': 2}, 'member_count_cap'),
    ({'max_member_bytes': 20}, 'member_bytes_cap'),
    ({'max_total_uncompressed_bytes': 20}, 'total_uncompressed_cap'),
    ({'max_json_bytes': 20}, 'json_bytes_cap'),
    ({'max_texture_side': 8}, 'texture_side_cap'),
    ({'max_total_rgba_bytes': 1000}, 'rgba_bytes_cap'),
    ({'max_faces': 5}, 'face_count_cap'),
    ({'max_work': 1}, 'work_cap'),
])
def test_caps_abort_entire_load(caps, reason):
    with pytest.raises(CatalogueError, match=reason):
        load(**caps)


def test_aggregate_rgba_cap_counts_multiple_textures_once_and_aborts():
    items = assets()
    other = 'assets/minecraft/textures/block/other.png'
    items[other] = png()  # Same pixels, distinct retained texture identities.
    items[MODEL]['textures']['other'] = 'block/other'
    items[MODEL]['elements'][0]['faces']['east']['texture'] = '#other'
    # Each 16x16 RGBA costs 1024 bytes. The original texture is reused by
    # five faces; the second texture takes the retained total to 2048.
    result = load(items, max_total_rgba_bytes=2048)
    assert len(result.faces) == 6
    assert len({f.texture_id for f in result.faces}) == 2
    assert len({f.rgba_sha256 for f in result.faces}) == 1
    with pytest.raises(CatalogueError, match='^rgba_bytes_cap$') as error:
        load(items, max_total_rgba_bytes=2047)
    assert error.value.reason == 'rgba_bytes_cap'


def test_parent_depth_and_state_caps_abort():
    items = assets()
    items[MODEL]['parent'] = 'block/root'
    items['assets/minecraft/models/block/root.json'] = {}
    with pytest.raises(CatalogueError, match='parent_depth_cap'):
        load(items, max_parent_depth=1)
    items = assets()
    items[STATE]['variants'] = {'axis=x': {'model': 'block/cube'},
                               'axis=y': {'model': 'block/cube'}}
    with pytest.raises(CatalogueError, match='state_count_cap'):
        load(items, max_states=1)


@pytest.mark.parametrize('field', list(CataloguePolicy.__dataclass_fields__))
def test_policy_limits_cannot_be_disabled_or_increased(field):
    with pytest.raises(CatalogueError, match='invalid_policy'):
        replace(CataloguePolicy(), **{field: 0})
    with pytest.raises(CatalogueError, match='invalid_policy'):
        replace(CataloguePolicy(), **{field: True})
    with pytest.raises(CatalogueError, match='invalid_policy'):
        replace(CataloguePolicy(), **{field: getattr(CataloguePolicy(), field) + 1})


def test_bad_archives_sources_duplicates_and_unsafe_paths(tmp_path):
    for raw in (b'', b'notzip', archive({})):
        with pytest.raises(CatalogueError):
            load_unit_face_catalogue(raw)
    with pytest.raises(CatalogueError, match='unreadable_source'):
        load_unit_face_catalogue(tmp_path/'missing.zip')
    with pytest.raises(CatalogueError, match='invalid_source'):
        load_unit_face_catalogue(io.BytesIO(archive(assets())))
    with pytest.raises(CatalogueError, match='invalid_policy'):
        load_unit_face_catalogue(archive(assets()), {})
    for name in ('../escape', '/absolute'):
        items = assets()
        items[name] = b''
        with pytest.raises(CatalogueError, match='unsafe_member_path'):
            load(items)
    items = assets()
    items['assets/minecraft/foo'] = b''
    raw = archive(items).replace(b'assets/minecraft/foo', b'assets\\minecraft\\foo')
    with pytest.raises(CatalogueError, match='unsafe_member_path'):
        load_unit_face_catalogue(raw)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as zipped:
        zipped.writestr(STATE, '{}')
        with pytest.warns(UserWarning):
            zipped.writestr(STATE, '{}')
    with pytest.raises(CatalogueError, match='duplicate_member'):
        load_unit_face_catalogue(stream.getvalue())


def test_forged_central_directory_count_cannot_bypass_preallocation_cap():
    raw = bytearray(archive(assets()))
    eocd = raw.rfind(b'PK\x05\x06')
    struct.pack_into('<2H', raw, eocd+8, 1, 1)
    with pytest.raises(CatalogueError, match='member_count_cap'):
        load_unit_face_catalogue(bytes(raw), replace(CataloguePolicy(), max_members=2))
    with pytest.raises(CatalogueError, match='corrupt_archive'):
        load_unit_face_catalogue(bytes(raw))
