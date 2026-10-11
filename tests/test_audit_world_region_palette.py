"""Focused synthetic tests for the E7 read-only world-region palette scanner.

Every fixture here is a minimal synthetic ``.mca``/NBT world built in pytest's
temporary directory by this file's own encoder, so the scanner is exercised
against independently written bytes. No real save, Minecraft install, launcher,
screenshot, JAR or production file is read or written.
"""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import zlib

import nbtlib
import pytest
from nbtlib import tag

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/evaluation/audit_world_region_palette.py"
SPEC = importlib.util.spec_from_file_location("e7_world_palette_scanner", SCRIPT)
subject = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = subject
SPEC.loader.exec_module(subject)

SECTOR = 4096
FULL = "minecraft:full"
STONE = ("minecraft:stone", [])
OAK_X = ("minecraft:oak_log", [("axis", "x")])
OAK_Z = ("minecraft:oak_log", [("axis", "z")])


# --------------------------------------------------------------------------- #
# Synthetic Anvil/NBT fixture builders (independent of the scanner)
# --------------------------------------------------------------------------- #


def _section(y, states, *, data=None, block_states=True):
    if not block_states:
        return tag.Compound({"Y": tag.Byte(y)})
    palette = tag.List[tag.Compound]([
        tag.Compound({
            "Name": tag.String(name),
            **({"Properties": tag.Compound(
                {key: tag.String(value) for key, value in properties})} if properties else {}),
        })
        for name, properties in states
    ])
    payload = {"palette": palette}
    if data is not None:
        payload["data"] = tag.LongArray(data)
    return tag.Compound({"Y": tag.Byte(y), "block_states": tag.Compound(payload)})


def _pack(indices, palette_size):
    """Pack palette indexes without letting an entry span two longs."""
    bits = max(4, (palette_size - 1).bit_length())
    per_long = 64 // bits
    longs = []
    for start in range(0, len(indices), per_long):
        value = 0
        for offset, index in enumerate(indices[start:start + per_long]):
            value |= (index & ((1 << bits) - 1)) << (offset * bits)
        longs.append(value - (1 << 64) if value >= (1 << 63) else value)
    return longs


def _indices_for(counts):
    indices = []
    for index, count in enumerate(counts):
        indices.extend([index] * count)
    assert len(indices) == 4096, len(indices)
    return indices


def _chunk_bytes(sections, *, data_version=3465, status=FULL, x=None, z=None, level=False,
                 extra=None):
    payload = {"DataVersion": tag.Int(data_version), "Status": tag.String(status)}
    if x is not None:
        payload["xPos"] = tag.Int(x)
    if z is not None:
        payload["zPos"] = tag.Int(z)
    if level:
        payload["Level"] = tag.Compound({"Sections": tag.List[tag.Compound]([])})
    else:
        payload["sections"] = tag.List[tag.Compound](sections)
    if extra:
        payload.update(extra)
    buffer = io.BytesIO()
    nbtlib.File(tag.Compound(payload)).write(buffer, byteorder="big")
    return buffer.getvalue()


def _modified_utf8(text):
    """Encode Java modified UTF-8: NUL as C0 80, non-BMP as CESU-8 surrogates."""
    out = bytearray()
    for character in text:
        code = ord(character)
        if code == 0:
            out += b"\xc0\x80"
        elif code < 0x80:
            out.append(code)
        elif code < 0x800:
            out += bytes([0xC0 | (code >> 6), 0x80 | (code & 0x3F)])
        elif code < 0x10000:
            out += bytes([0xE0 | (code >> 12), 0x80 | ((code >> 6) & 0x3F),
                          0x80 | (code & 0x3F)])
        else:
            code -= 0x10000
            for unit in (0xD800 | (code >> 10), 0xDC00 | (code & 0x3FF)):
                out += bytes([0xE0 | (unit >> 12), 0x80 | ((unit >> 6) & 0x3F),
                              0x80 | (unit & 0x3F)])
    return bytes(out)


def _rename_palette_name(chunk, placeholder, raw):
    """Replace the palette's placeholder string with raw modified-UTF-8 bytes."""
    needle = len(placeholder).to_bytes(2, "big") + placeholder.encode("utf-8")
    assert chunk.count(needle) == 1, chunk.count(needle)
    return chunk.replace(needle, len(raw).to_bytes(2, "big") + raw)


def _write_region(path, chunks, *, compression=2, version=None):
    """Write a region file. ``chunks`` maps (cx, cz) to raw chunk NBT bytes."""
    locations = bytearray(SECTOR)
    timestamps = bytearray(SECTOR)
    body = bytearray()
    next_sector = 2
    for (chunk_x, chunk_z), payload in sorted(chunks.items()):
        if compression == 3:
            stored = payload
        elif compression == 1:
            stored = gzip.compress(payload)
        else:
            stored = zlib.compress(payload)
        number = compression if version is None else version
        framed = (len(stored) + 1).to_bytes(4, "big") + bytes([number]) + stored
        used = max(1, math.ceil(len(framed) / SECTOR))
        framed += b"\x00" * (used * SECTOR - len(framed))
        slot = ((chunk_x & 31) + (chunk_z & 31) * 32) * 4
        locations[slot:slot + 3] = next_sector.to_bytes(3, "big")
        locations[slot + 3] = used
        timestamps[slot:slot + 4] = b"\x00\x00\x00\x01"
        body += framed
        next_sector += used
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(locations) + bytes(timestamps) + bytes(body))
    return path


def _region_path(game, save, region_x, region_z):
    return game / "saves" / save / "region" / f"r.{region_x}.{region_z}.mca"


def _chunk_bytes_exact_size(sections, target, **kwargs):
    """Chunk NBT padded with a byte array so its encoding is exactly ``target`` bytes."""
    base = len(_chunk_bytes(sections, **kwargs))
    name = "padding"
    # tag id (1) + name length (2) + name + int-array length (4)
    overhead = 1 + 2 + len(name) + 4
    padding = target - base - overhead
    assert padding >= 0, f"target {target} is below the {base}-byte base chunk"
    raw = _chunk_bytes(sections, extra={name: tag.ByteArray([0] * padding)}, **kwargs)
    assert len(raw) == target, (len(raw), target)
    return raw


def _stored_payload_length(region_path):
    """Read the declared payload length of the first chunk frame from a written region file."""
    raw = region_path.read_bytes()
    declared = int.from_bytes(raw[2 * SECTOR:2 * SECTOR + 4], "big")
    return declared - 1


def _game(tmp_path, save="Alpha"):
    game = tmp_path / "game"
    (game / "saves" / save).mkdir(parents=True, exist_ok=True)
    return game


def _region(save, minimum, maximum, **extra):
    payload = {"save": save, "min": list(minimum), "max": list(maximum)}
    payload.update(extra)
    return payload


def _write_request(tmp_path, regions, name="request.json", **extra):
    payload = {"schema": subject.REQUEST_SCHEMA, "regions": regions}
    payload.update(extra)
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _run(game, request_path):
    stream = io.StringIO()
    code = subject.main(
        ["--game-dir", str(game), "--regions-json", str(request_path)], stdout=stream)
    return code, stream.getvalue()


def _scan(tmp_path, game, regions, name="request.json"):
    return _run(game, _write_request(tmp_path, regions, name))


def _document(tmp_path, game, regions, name="request.json"):
    code, text = _scan(tmp_path, game, regions, name)
    assert code == 0, text
    return json.loads(text)


def _refusal(tmp_path, game, regions, name="request.json"):
    code, text = _scan(tmp_path, game, regions, name)
    assert code == 2, text
    document = json.loads(text)
    assert document["status"] == "refused"
    return document["error"]["message"]


def _patch(path, start, data):
    raw = bytearray(path.read_bytes())
    raw[start:start + len(data)] = data
    path.write_bytes(bytes(raw))


def _snapshot(root):
    entries = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            entries[relative] = ("dir",)
        else:
            entries[relative] = (
                "file", path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
    return entries


# --------------------------------------------------------------------------- #
# Exact counts and canonicalization
# --------------------------------------------------------------------------- #


def test_exact_counts_and_canonical_property_order(tmp_path):
    game = _game(tmp_path)
    palette = [
        STONE,
        OAK_Z,
        OAK_X,
        ("minecraft:cave_air", []),
        ("minecraft:oak_fence", [("west", "true"), ("east", "false"), ("waterlogged", "false")]),
    ]
    indices = [1, 2, 2, 3, 4] + [0] * 4091
    _write_region(
        _region_path(game, "Alpha", 0, 0),
        {(0, 0): _chunk_bytes([_section(0, palette, data=_pack(indices, len(palette)))], x=0, z=0)},
    )
    region = _document(tmp_path, game, [_region("Alpha", (0, 0, 0), (16, 16, 16))])["regions"][0]

    assert region["non_air_total"] == 4095
    assert region["air_total"] == 1
    assert region["scanned_block_positions"] == 4096
    assert region["state_counts"] == {
        "minecraft:oak_fence[east=false,waterlogged=false,west=true]": 1,
        "minecraft:oak_log[axis=x]": 2,
        "minecraft:oak_log[axis=z]": 1,
        "minecraft:stone": 4091,
    }


def test_air_family_states_are_aggregated_not_named(tmp_path):
    game = _game(tmp_path)
    palette = [STONE, ("minecraft:air", []), ("minecraft:cave_air", []), ("minecraft:void_air", [])]
    indices = [1, 2, 3] + [0] * 4093
    _write_region(
        _region_path(game, "Alpha", 0, 0),
        {(0, 0): _chunk_bytes([_section(0, palette, data=_pack(indices, len(palette)))])},
    )
    region = _document(tmp_path, game, [_region("Alpha", (0, 0, 0), (16, 16, 16))])["regions"][0]
    assert region["air_total"] == 3
    assert region["non_air_total"] == 4093
    assert set(region["state_counts"]) == {"minecraft:stone"}


def test_singleton_palette_without_a_data_array(tmp_path):
    game = _game(tmp_path)
    _write_region(
        _region_path(game, "Alpha", 0, 0),
        {(0, 0): _chunk_bytes([_section(0, [STONE])])},
    )
    region = _document(tmp_path, game, [_region("Alpha", (0, 0, 0), (16, 16, 16))])["regions"][0]
    assert region["state_counts"] == {"minecraft:stone": 4096}
    assert region["non_air_total"] == 4096
    assert region["air_total"] == 0


def test_packed_palette_wider_than_four_bits_and_partial_y_window(tmp_path):
    game = _game(tmp_path)
    palette = [(f"minecraft:test_block_{index}", []) for index in range(20)]
    indices = [index % 20 for index in range(4096)]
    _write_region(
        _region_path(game, "Alpha", 0, 0),
        {(0, 0): _chunk_bytes([_section(0, palette, data=_pack(indices, len(palette)))])},
    )
    region = _document(tmp_path, game, [_region("Alpha", (0, 0, 0), (16, 8, 16))])["regions"][0]
    assert region["scanned_block_positions"] == 16 * 8 * 16
    assert region["state_counts"]["minecraft:test_block_0"] == 103
    assert region["state_counts"]["minecraft:test_block_7"] == 103
    assert region["state_counts"]["minecraft:test_block_8"] == 102
    assert region["state_counts"]["minecraft:test_block_19"] == 102
    assert region["non_air_total"] == 2048


def test_empty_section_without_block_states_is_all_air(tmp_path):
    game = _game(tmp_path)
    _write_region(
        _region_path(game, "Alpha", 0, 0),
        {(0, 0): _chunk_bytes([_section(0, [], block_states=False)])},
    )
    region = _document(tmp_path, game, [_region("Alpha", (0, 0, 0), (16, 16, 16))])["regions"][0]
    assert region["state_counts"] == {}
    assert region["air_total"] == 4096
    assert region["sections_present"] == 1
    assert region["missing_sections"] == 0


def test_missing_section_is_counted_once_per_requested_span(tmp_path):
    game = _game(tmp_path)
    _write_region(
        _region_path(game, "Alpha", 0, 0),
        {(0, 0): _chunk_bytes([_section(0, [STONE])])},
    )
    region = _document(tmp_path, game, [_region("Alpha", (0, -16, 0), (16, 16, 16))])["regions"][0]
    assert region["sections_requested"] == 2
    assert region["sections_present"] == 1
    assert region["missing_sections"] == 1
    assert region["non_air_total"] == 4096


def test_data_version_2865_with_minecraft_full_status_is_accepted(tmp_path):
    """1.18 release (DataVersion 2865) writes Status='minecraft:full'."""
    game = _game(tmp_path)
    _write_region(
        _region_path(game, "Alpha", 0, 0),
        {(0, 0): _chunk_bytes([_section(0, [STONE])], data_version=2865)},
    )
    region = _document(tmp_path, game, [_region("Alpha", (0, 0, 0), (16, 16, 16))])["regions"][0]
    assert region["non_air_total"] == 4096
    assert region["data_versions"] == [2865]


def test_delimiter_characters_in_state_keys_do_not_collide(tmp_path):
    """Naive rendering would merge these two distinct states into one key."""
    game = _game(tmp_path)
    palette = [
        ("minecraft:test", [("a", "b,c=d")]),
        ("minecraft:test", [("a", "b"), ("c", "d")]),
    ]
    indices = [0] * 2048 + [1] * 2048
    _write_region(
        _region_path(game, "Alpha", 0, 0),
        {(0, 0): _chunk_bytes([_section(0, palette, data=_pack(indices, 2))], x=0, z=0)},
    )
    region = _document(tmp_path, game, [_region("Alpha", (0, 0, 0), (16, 16, 16))])["regions"][0]
    assert region["state_counts"] == {
        "minecraft:test[a=b,c=d]": 2048,
        "minecraft:test[a=b\\,c\\=d]": 2048,
    }
    assert region["non_air_total"] == 4096


def test_modified_utf8_nul_name_is_decoded(tmp_path):
    game = _game(tmp_path)
    placeholder = "PLACEHOLDERNAME"
    chunk = _chunk_bytes([_section(0, [(placeholder, [])])], x=0, z=0)
    name = "minecraft:nul\x00block"
    chunk = _rename_palette_name(chunk, placeholder, _modified_utf8(name))
    _write_region(_region_path(game, "Alpha", 0, 0), {(0, 0): chunk})
    region = _document(tmp_path, game, [_region("Alpha", (0, 0, 0), (16, 16, 16))])["regions"][0]
    assert region["state_counts"] == {name: 4096}


def test_modified_utf8_surrogate_pair_name_is_decoded(tmp_path):
    game = _game(tmp_path)
    placeholder = "PLACEHOLDERNAME"
    chunk = _chunk_bytes([_section(0, [(placeholder, [])])], x=0, z=0)
    name = "minecraft:grin\U0001F600"
    chunk = _rename_palette_name(chunk, placeholder, _modified_utf8(name))
    _write_region(_region_path(game, "Alpha", 0, 0), {(0, 0): chunk})
    region = _document(tmp_path, game, [_region("Alpha", (0, 0, 0), (16, 16, 16))])["regions"][0]
    assert region["state_counts"] == {name: 4096}


@pytest.mark.parametrize("raw, fragment", [
    (b"minecraft:bad\xc3", "ends inside a modified UTF-8 sequence"),
    (b"minecraft:bad\xc3\x28", "malformed modified UTF-8 continuation"),
    (b"minecraft:bad\xc1\x81", "overlong modified UTF-8"),
    (b"minecraft:bad\xf0\x9f\x98\x80", "not valid Java modified UTF-8"),
    (b"minecraft:bad\xed\xa0\xbd", "unpaired UTF-16 surrogate"),
    (b"minecraft:bad\xed\xb8\x80", "unpaired UTF-16 surrogate"),
])
def test_malformed_modified_utf8_string_is_refused(tmp_path, raw, fragment):
    game = _game(tmp_path)
    placeholder = "PLACEHOLDERNAME"
    chunk = _chunk_bytes([_section(0, [(placeholder, [])])], x=0, z=0)
    chunk = _rename_palette_name(chunk, placeholder, raw)
    _write_region(_region_path(game, "Alpha", 0, 0), {(0, 0): chunk})
    message = _refusal(tmp_path, game, [_region("Alpha", (0, 0, 0), (16, 16, 16))])
    assert fragment in message


# --------------------------------------------------------------------------- #
# Boundary coordinates
# --------------------------------------------------------------------------- #


def test_negative_chunk_and_region_coordinates(tmp_path):
    game = _game(tmp_path)
    _write_region(
        _region_path(game, "Alpha", -1, -1),
        {(-1, -1): _chunk_bytes([_section(0, [STONE])], x=-1, z=-1)},
    )
    region = _document(tmp_path, game, [_region("Alpha", (-16, 0, -16), (0, 8, 0))])["regions"][0]
    assert region["scanned_chunks"] == 1
    assert region["non_air_total"] == 16 * 8 * 16


def test_exclusive_max_does_not_pull_in_the_next_chunk(tmp_path):
    game = _game(tmp_path)
    _write_region(
        _region_path(game, "Alpha", 0, 0),
        {(0, 0): _chunk_bytes([_section(0, [STONE])], x=0, z=0)},
    )
    region = _document(tmp_path, game, [_region("Alpha", (0, 0, 0), (16, 16, 16))])["regions"][0]
    assert region["requested_chunks"] == 1
    assert region["scanned_block_positions"] == 4096


def test_region_spanning_two_region_files(tmp_path):
    game = _game(tmp_path)
    _write_region(
        _region_path(game, "Alpha", 0, 0),
        {(31, 0): _chunk_bytes([_section(0, [STONE])], x=31, z=0)},
    )
    _write_region(
        _region_path(game, "Alpha", 1, 0),
        {(32, 0): _chunk_bytes([_section(0, [STONE])], x=32, z=0)},
    )
    region = _document(
        tmp_path, game, [_region("Alpha", (496, 0, 0), (528, 8, 16))])["regions"][0]
    assert region["requested_chunks"] == 2
    assert region["scanned_chunks"] == 2
    assert region["missing_chunks"] == 0
    assert region["non_air_total"] == 16 * 8 * 16 * 2


def test_missing_chunks_are_counted_not_refused(tmp_path):
    game = _game(tmp_path)
    _write_region(
        _region_path(game, "Alpha", 0, 0),
        {(0, 0): _chunk_bytes([_section(0, [STONE])], x=0, z=0)},
    )
    region = _document(
        tmp_path, game, [_region("Alpha", (0, 0, 0), (32, 16, 32))])["regions"][0]
    assert region["requested_chunks"] == 4
    assert region["scanned_chunks"] == 1
    assert region["missing_chunks"] == 3


def test_missing_region_file_is_refused(tmp_path):
    game = _game(tmp_path)
    (game / "saves" / "Alpha" / "region").mkdir(parents=True)
    message = _refusal(tmp_path, game, [_region("Alpha", (512, 0, 0), (528, 8, 16))])
    assert "missing required region file" in message


def test_missing_region_directory_is_refused(tmp_path):
    game = _game(tmp_path)
    message = _refusal(tmp_path, game, [_region("Alpha", (0, 0, 0), (16, 16, 16))])
    assert "no region directory" in message


# --------------------------------------------------------------------------- #
# Compression support and explicit bounds
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("compression", [1, 2, 3])
def test_supported_compression_types_decode(tmp_path, compression):
    game = _game(tmp_path, save=f"World{compression}")
    _write_region(
        _region_path(game, f"World{compression}", 0, 0),
        {(0, 0): _chunk_bytes([_section(0, [STONE])], x=0, z=0)},
        compression=compression,
    )
    region = _document(
        tmp_path, game, [_region(f"World{compression}", (0, 0, 0), (16, 16, 16))],
        name=f"request-{compression}.json")["regions"][0]
    assert region["non_air_total"] == 4096


def test_external_mcc_chunk_is_refused(tmp_path):
    game = _game(tmp_path)
    _write_region(
        _region_path(game, "Alpha", 0, 0),
        {(0, 0): _chunk_bytes([_section(0, [STONE])])},
        version=2 | 0x80,
    )
    message = _refusal(tmp_path, game, [_region("Alpha", (0, 0, 0), (16, 16, 16))])
    assert ".mcc" in message


def test_lz4_and_unknown_compression_types_are_refused(tmp_path):
    game = _game(tmp_path)
    _write_region(
        _region_path(game, "Alpha", 0, 0),
        {(0, 0): _chunk_bytes([_section(0, [STONE])])},
        version=4,
    )
    message = _refusal(tmp_path, game, [_region("Alpha", (0, 0, 0), (16, 16, 16))])
    assert "unsupported chunk compression type 4" in message


def test_compressed_chunk_byte_bound_is_enforced(tmp_path, monkeypatch):
    game = _game(tmp_path)
    _write_region(
        _region_path(game, "Alpha", 0, 0),
        {(0, 0): _chunk_bytes([_section(0, [STONE])])},
    )
    monkeypatch.setattr(subject, "MAX_COMPRESSED_CHUNK_BYTES", 8)
    message = _refusal(tmp_path, game, [_region("Alpha", (0, 0, 0), (16, 16, 16))])
    assert "compressed chunk frame exceeds the byte bound" in message


@pytest.mark.parametrize("compression", [2, 3])
def test_decompressed_chunk_byte_bound_is_enforced(tmp_path, monkeypatch, compression):
    game = _game(tmp_path, save=f"Bound{compression}")
    _write_region(
        _region_path(game, f"Bound{compression}", 0, 0),
        {(0, 0): _chunk_bytes([_section(0, [STONE])])},
        compression=compression,
    )
    monkeypatch.setattr(subject, "MAX_DECOMPRESSED_CHUNK_BYTES", 64)
    message = _refusal(
        tmp_path, game, [_region(f"Bound{compression}", (0, 0, 0), (16, 16, 16))],
        name=f"bound-{compression}.json")
    assert "decompressed byte bound" in message


@pytest.mark.parametrize("compression", [1, 2])
def test_decompressed_chunk_byte_bound_accepts_exactly_the_limit(
        tmp_path, monkeypatch, compression):
    """A gzip/zlib chunk decompressing to exactly the cap is accepted."""
    save = f"Exact{compression}"
    game = _game(tmp_path, save=save)
    limit = 512
    chunk = _chunk_bytes_exact_size([_section(0, [STONE])], limit)
    _write_region(
        _region_path(game, save, 0, 0), {(0, 0): chunk}, compression=compression)
    monkeypatch.setattr(subject, "MAX_DECOMPRESSED_CHUNK_BYTES", limit)
    code, text = _scan(
        tmp_path, game, [_region(save, (0, 0, 0), (16, 16, 16))],
        name=f"exact-decompressed-{compression}.json")
    assert code == 0, text
    assert json.loads(text)["regions"][0]["non_air_total"] == 4096


@pytest.mark.parametrize("compression", [1, 2])
def test_decompressed_chunk_byte_bound_refuses_one_byte_over(
        tmp_path, monkeypatch, compression):
    save = f"Over{compression}"
    game = _game(tmp_path, save=save)
    limit = 512
    chunk = _chunk_bytes_exact_size([_section(0, [STONE])], limit + 1)
    _write_region(
        _region_path(game, save, 0, 0), {(0, 0): chunk}, compression=compression)
    monkeypatch.setattr(subject, "MAX_DECOMPRESSED_CHUNK_BYTES", limit)
    message = _refusal(
        tmp_path, game, [_region(save, (0, 0, 0), (16, 16, 16))],
        name=f"over-decompressed-{compression}.json")
    assert "decompressed byte bound" in message


@pytest.mark.parametrize("compression", [1, 2])
def test_compressed_chunk_byte_bound_accepts_exactly_the_limit(
        tmp_path, monkeypatch, compression):
    """A frame whose compressed payload is exactly the cap is accepted."""
    save = f"Packed{compression}"
    game = _game(tmp_path, save=save)
    region = _write_region(
        _region_path(game, save, 0, 0),
        {(0, 0): _chunk_bytes([_section(0, [STONE])])},
        compression=compression,
    )
    monkeypatch.setattr(subject, "MAX_COMPRESSED_CHUNK_BYTES", _stored_payload_length(region))
    code, text = _scan(
        tmp_path, game, [_region(save, (0, 0, 0), (16, 16, 16))],
        name=f"exact-compressed-{compression}.json")
    assert code == 0, text


@pytest.mark.parametrize("compression", [1, 2])
def test_compressed_chunk_byte_bound_refuses_one_byte_over(
        tmp_path, monkeypatch, compression):
    save = f"Tight{compression}"
    game = _game(tmp_path, save=save)
    region = _write_region(
        _region_path(game, save, 0, 0),
        {(0, 0): _chunk_bytes([_section(0, [STONE])])},
        compression=compression,
    )
    actual = _stored_payload_length(region)
    monkeypatch.setattr(subject, "MAX_COMPRESSED_CHUNK_BYTES", actual - 1)
    message = _refusal(
        tmp_path, game, [_region(save, (0, 0, 0), (16, 16, 16))],
        name=f"over-compressed-{compression}.json")
    assert "compressed chunk frame exceeds the byte bound" in message


def test_nbt_depth_budget_is_enforced(tmp_path):
    game = _game(tmp_path)
    deep = tag.Compound({})
    node = deep
    for _ in range(30):
        child = tag.Compound({})
        node["nest"] = child
        node = child
    _write_region(
        _region_path(game, "Alpha", 0, 0),
        {(0, 0): _chunk_bytes([_section(0, [STONE])], extra={"Deep": deep})},
    )
    message = _refusal(tmp_path, game, [_region("Alpha", (0, 0, 0), (16, 16, 16))])
    assert "depth bound" in message


def test_nbt_element_budget_is_enforced(tmp_path, monkeypatch):
    game = _game(tmp_path)
    _write_region(
        _region_path(game, "Alpha", 0, 0),
        {(0, 0): _chunk_bytes([_section(0, [STONE])])},
    )
    monkeypatch.setattr(subject, "MAX_NBT_ELEMENTS", 1)
    message = _refusal(tmp_path, game, [_region("Alpha", (0, 0, 0), (16, 16, 16))])
    assert "element budget" in message


def test_nbt_array_length_budget_is_enforced(tmp_path, monkeypatch):
    game = _game(tmp_path)
    _write_region(
        _region_path(game, "Alpha", 0, 0),
        {(0, 0): _chunk_bytes([_section(0, [STONE, OAK_X], data=[0] * 256)])},
    )
    monkeypatch.setattr(subject, "MAX_NBT_ELEMENTS", 100)
    message = _refusal(tmp_path, game, [_region("Alpha", (0, 0, 0), (16, 16, 16))])
    assert "array length is outside the element budget" in message


# --------------------------------------------------------------------------- #
# Corrupt / unsupported data
# --------------------------------------------------------------------------- #


def test_sector_offset_beyond_the_file_is_refused(tmp_path):
    game = _game(tmp_path)
    region_file = _write_region(
        _region_path(game, "Alpha", 0, 0),
        {(0, 0): _chunk_bytes([_section(0, [STONE])])},
    )
    _patch(region_file, 0, (99999).to_bytes(3, "big"))
    message = _refusal(tmp_path, game, [_region("Alpha", (0, 0, 0), (16, 16, 16))])
    assert "beyond the region file" in message


def test_sector_offset_into_the_header_is_refused(tmp_path):
    game = _game(tmp_path)
    region_file = _write_region(
        _region_path(game, "Alpha", 0, 0),
        {(0, 0): _chunk_bytes([_section(0, [STONE])])},
    )
    _patch(region_file, 0, (1).to_bytes(3, "big"))
    message = _refusal(tmp_path, game, [_region("Alpha", (0, 0, 0), (16, 16, 16))])
    assert "points into the region header" in message


def test_frame_overrunning_its_allocation_is_refused(tmp_path):
    game = _game(tmp_path)
    region_file = _write_region(
        _region_path(game, "Alpha", 0, 0),
        {(0, 0): _chunk_bytes([_section(0, [STONE])])},
    )
    _patch(region_file, 2 * SECTOR, (5000).to_bytes(4, "big"))
    message = _refusal(tmp_path, game, [_region("Alpha", (0, 0, 0), (16, 16, 16))])
    assert "overflows its allocated sectors" in message


def test_truncated_compressed_stream_is_refused(tmp_path):
    game = _game(tmp_path)
    region_file = _write_region(
        _region_path(game, "Alpha", 0, 0),
        {(0, 0): _chunk_bytes([_section(0, [STONE])])},
    )
    declared = int.from_bytes(region_file.read_bytes()[2 * SECTOR:2 * SECTOR + 4], "big")
    _patch(region_file, 2 * SECTOR, (declared - 4).to_bytes(4, "big"))
    message = _refusal(tmp_path, game, [_region("Alpha", (0, 0, 0), (16, 16, 16))])
    assert "compressed stream is truncated" in message


def test_truncated_region_file_is_refused(tmp_path):
    game = _game(tmp_path)
    region_file = _write_region(
        _region_path(game, "Alpha", 0, 0),
        {(0, 0): _chunk_bytes([_section(0, [STONE])])},
    )
    region_file.write_bytes(region_file.read_bytes()[: 2 * SECTOR + 16])
    message = _refusal(tmp_path, game, [_region("Alpha", (0, 0, 0), (16, 16, 16))])
    assert "extends beyond the region file" in message


def test_payload_fits_but_allocation_past_eof_is_refused(tmp_path):
    """The declared payload fits its one real sector, but the header claims more."""
    game = _game(tmp_path)
    region_file = _write_region(
        _region_path(game, "Alpha", 0, 0),
        {(0, 0): _chunk_bytes([_section(0, [STONE])], x=0, z=0)},
    )
    declared = int.from_bytes(region_file.read_bytes()[2 * SECTOR:2 * SECTOR + 4], "big")
    assert 5 + declared - 1 <= SECTOR  # the payload really does fit one sector
    _patch(region_file, 3, b"\x03")  # claim three sectors; only one exists
    message = _refusal(tmp_path, game, [_region("Alpha", (0, 0, 0), (16, 16, 16))])
    assert "extends beyond the region file" in message


@pytest.mark.parametrize("offset, count", [(0, 1), (2, 0)])
def test_half_zero_location_entry_is_refused(tmp_path, offset, count):
    """Exactly one of offset/sector count set is malformed, even out of selection."""
    game = _game(tmp_path)
    region_file = _write_region(
        _region_path(game, "Alpha", 0, 0),
        {(0, 0): _chunk_bytes([_section(0, [STONE])])},
    )
    base = 5 * 4  # an unselected slot
    if offset:
        _patch(region_file, base, offset.to_bytes(3, "big"))
    if count:
        _patch(region_file, base + 3, bytes([count]))
    message = _refusal(tmp_path, game, [_region("Alpha", (0, 0, 0), (16, 16, 16))])
    assert "half-zero" in message


def test_overlap_with_an_unselected_occupied_slot_is_refused(tmp_path):
    """Overlap is found from the header; the unselected payload is never read."""
    game = _game(tmp_path)
    region_file = _write_region(
        _region_path(game, "Alpha", 0, 0),
        {
            (0, 0): _chunk_bytes([_section(0, [STONE])], x=0, z=0),
            (1, 0): _chunk_bytes([_section(0, [STONE])], x=1, z=0),
        },
    )
    # Point chunk (1, 0)'s allocation at chunk (0, 0)'s sectors.
    _patch(region_file, 4, (2).to_bytes(3, "big"))
    # If that payload were ever read it would refuse as unsupported compression.
    _patch(region_file, 3 * SECTOR + 4, b"\x04")
    message = _refusal(tmp_path, game, [_region("Alpha", (0, 0, 0), (16, 16, 16))])
    assert "overlap" in message


def test_trailing_bytes_after_the_nbt_root_are_refused(tmp_path):
    game = _game(tmp_path)
    _write_region(
        _region_path(game, "Alpha", 0, 0),
        {(0, 0): _chunk_bytes([_section(0, [STONE])]) + b"\x00"},
    )
    message = _refusal(tmp_path, game, [_region("Alpha", (0, 0, 0), (16, 16, 16))])
    assert "trailing bytes" in message


def test_invalid_palette_index_is_refused(tmp_path):
    game = _game(tmp_path)
    indices = [5] + [0] * 4095
    _write_region(
        _region_path(game, "Alpha", 0, 0),
        {(0, 0): _chunk_bytes([_section(0, [STONE, OAK_X], data=_pack(indices, 2))])},
    )
    message = _refusal(tmp_path, game, [_region("Alpha", (0, 0, 0), (16, 16, 16))])
    assert "palette index is out of range" in message


def test_wrong_data_array_length_is_refused(tmp_path):
    game = _game(tmp_path)
    _write_region(
        _region_path(game, "Alpha", 0, 0),
        {(0, 0): _chunk_bytes([_section(0, [STONE, OAK_X], data=[0])])},
    )
    message = _refusal(tmp_path, game, [_region("Alpha", (0, 0, 0), (16, 16, 16))])
    assert "expected 256" in message


def test_multi_entry_palette_without_data_is_refused(tmp_path):
    game = _game(tmp_path)
    _write_region(
        _region_path(game, "Alpha", 0, 0),
        {(0, 0): _chunk_bytes([_section(0, [STONE, OAK_X])])},
    )
    message = _refusal(tmp_path, game, [_region("Alpha", (0, 0, 0), (16, 16, 16))])
    assert "no data array" in message


def test_non_long_data_array_is_refused(tmp_path):
    game = _game(tmp_path)
    section = tag.Compound({
        "Y": tag.Byte(0),
        "block_states": tag.Compound({
            "palette": tag.List[tag.Compound]([
                tag.Compound({"Name": tag.String("minecraft:stone")}),
                tag.Compound({"Name": tag.String("minecraft:dirt")}),
            ]),
            "data": tag.IntArray([0] * 256),
        }),
    })
    _write_region(
        _region_path(game, "Alpha", 0, 0),
        {(0, 0): _chunk_bytes([section])},
    )
    message = _refusal(tmp_path, game, [_region("Alpha", (0, 0, 0), (16, 16, 16))])
    assert "not a long array" in message


def test_pre_118_level_layout_is_refused(tmp_path):
    game = _game(tmp_path)
    _write_region(
        _region_path(game, "Alpha", 0, 0),
        {(0, 0): _chunk_bytes([], level=True)},
    )
    message = _refusal(tmp_path, game, [_region("Alpha", (0, 0, 0), (16, 16, 16))])
    assert "pre-1.18" in message


def test_out_of_range_data_version_is_refused(tmp_path):
    game = _game(tmp_path)
    _write_region(
        _region_path(game, "Alpha", 0, 0),
        {(0, 0): _chunk_bytes([_section(0, [STONE])], data_version=2000)},
    )
    message = _refusal(tmp_path, game, [_region("Alpha", (0, 0, 0), (16, 16, 16))])
    assert "DataVersion" in message


def test_partially_generated_chunk_is_refused(tmp_path):
    game = _game(tmp_path)
    _write_region(
        _region_path(game, "Alpha", 0, 0),
        {(0, 0): _chunk_bytes([_section(0, [STONE])], status="minecraft:features")},
    )
    message = _refusal(tmp_path, game, [_region("Alpha", (0, 0, 0), (16, 16, 16))])
    assert "not fully generated" in message


def test_chunk_position_mismatch_is_refused(tmp_path):
    game = _game(tmp_path)
    _write_region(
        _region_path(game, "Alpha", 0, 0),
        {(0, 0): _chunk_bytes([_section(0, [STONE])], x=7, z=0)},
    )
    message = _refusal(tmp_path, game, [_region("Alpha", (0, 0, 0), (16, 16, 16))])
    assert "xPos" in message


# --------------------------------------------------------------------------- #
# Request and path safety
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("save", [
    "../outside", "sub/dir", "..", ".", "C:\\absolute", "C:", "trailing.", " leading",
    "back\\slash", "nul\0name", "",
])
def test_unsafe_save_names_are_refused(tmp_path, save):
    game = _game(tmp_path)
    message = _refusal(tmp_path, game, [_region(save, (0, 0, 0), (16, 16, 16))])
    assert "save directory name" in message


def test_link_like_save_directory_is_refused(tmp_path, monkeypatch):
    """The guard is exercised without creating a reparse point on this host."""
    game = _game(tmp_path)
    save_root = game / "saves" / "Linked"
    save_root.mkdir(parents=True)
    real = subject.is_link_like
    monkeypatch.setattr(
        subject, "is_link_like",
        lambda path: True if Path(path) == save_root else real(path))
    message = _refusal(tmp_path, game, [_region("Linked", (0, 0, 0), (16, 16, 16))])
    assert "link-like" in message


def test_link_like_region_file_is_refused(tmp_path, monkeypatch):
    game = _game(tmp_path)
    region_file = _write_region(
        _region_path(game, "Alpha", 0, 0),
        {(0, 0): _chunk_bytes([_section(0, [STONE])])},
    )
    real = subject.is_link_like
    monkeypatch.setattr(
        subject, "is_link_like",
        lambda path: True if Path(path) == region_file else real(path))
    message = _refusal(tmp_path, game, [_region("Alpha", (0, 0, 0), (16, 16, 16))])
    assert "link-like" in message


def test_link_like_game_directory_is_refused(tmp_path, monkeypatch):
    game = _game(tmp_path)
    real = subject.is_link_like
    monkeypatch.setattr(
        subject, "is_link_like",
        lambda path: True if Path(path) == game else real(path))
    message = _refusal(tmp_path, game, [_region("Alpha", (0, 0, 0), (16, 16, 16))])
    assert "link-like" in message


def test_link_like_saves_root_is_refused(tmp_path, monkeypatch):
    game = _game(tmp_path)
    saves_root = game / "saves"
    real = subject.is_link_like
    monkeypatch.setattr(
        subject, "is_link_like",
        lambda path: True if Path(path) == saves_root else real(path))
    message = _refusal(tmp_path, game, [_region("Alpha", (0, 0, 0), (16, 16, 16))])
    assert "link-like" in message


def test_link_like_intermediate_directory_is_refused(tmp_path, monkeypatch):
    """An ancestor link would redirect the whole read; it is refused too."""
    intermediate = tmp_path / "linked"
    game = intermediate / "game"
    (game / "saves" / "Alpha").mkdir(parents=True)
    real = subject.is_link_like
    monkeypatch.setattr(
        subject, "is_link_like",
        lambda path: True if Path(path) == intermediate else real(path))
    message = _refusal(tmp_path, game, [_region("Alpha", (0, 0, 0), (16, 16, 16))])
    assert "link-like" in message


@pytest.mark.parametrize("relative", ["game", "."])
def test_link_like_cwd_ancestor_is_refused_for_a_relative_game_dir(
        tmp_path, monkeypatch, relative):
    """A relative --game-dir must check the lexically-absolute cwd ancestors."""
    _game(tmp_path)
    request = _write_request(tmp_path, [_region("Alpha", (0, 0, 0), (16, 16, 16))])
    monkeypatch.chdir(tmp_path)
    real = subject.is_link_like
    monkeypatch.setattr(
        subject, "is_link_like",
        lambda path: True if Path(path) == tmp_path else real(path))
    code, text = _run(relative, request)
    assert code == 2, text
    assert "link-like" in json.loads(text)["error"]["message"]


def test_unknown_request_keys_are_refused(tmp_path):
    game = _game(tmp_path)
    path = _write_request(tmp_path, [_region("Alpha", (0, 0, 0), (16, 16, 16))])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["extra"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    code, text = _run(game, path)
    assert code == 2
    assert "unsupported key" in json.loads(text)["error"]["message"]


def test_unknown_region_keys_are_refused(tmp_path):
    game = _game(tmp_path)
    message = _refusal(
        tmp_path, game, [_region("Alpha", (0, 0, 0), (16, 16, 16), dimension="nether")])
    assert "unsupported key" in message


def test_boolean_coordinate_is_refused(tmp_path):
    game = _game(tmp_path)
    message = _refusal(
        tmp_path, game, [{"save": "Alpha", "min": [True, 0, 0], "max": [16, 16, 16]}])
    assert "integers" in message


def test_wrong_request_schema_is_refused(tmp_path):
    game = _game(tmp_path)
    path = _write_request(tmp_path, [_region("Alpha", (0, 0, 0), (16, 16, 16))])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema"] = "wrong"
    path.write_text(json.dumps(payload), encoding="utf-8")
    code, text = _run(game, path)
    assert code == 2
    assert "schema" in json.loads(text)["error"]["message"]


def test_empty_region_list_is_refused(tmp_path):
    game = _game(tmp_path)
    message = _refusal(tmp_path, game, [])
    assert "non-empty" in message


def test_region_edge_cap_is_enforced(tmp_path):
    game = _game(tmp_path)
    message = _refusal(tmp_path, game, [_region("Alpha", (0, 0, 0), (513, 1, 1))])
    assert "MAX_REGION_EDGE" in message


def test_region_volume_cap_is_enforced(tmp_path):
    game = _game(tmp_path)
    message = _refusal(tmp_path, game, [_region("Alpha", (0, 0, 0), (512, 1, 512))])
    assert "MAX_REGION_VOLUME" in message


def test_inverted_region_is_refused(tmp_path):
    game = _game(tmp_path)
    message = _refusal(tmp_path, game, [_region("Alpha", (16, 0, 0), (0, 16, 16))])
    assert "max must exceed min" in message


def test_unknown_save_is_refused(tmp_path):
    game = _game(tmp_path)
    message = _refusal(tmp_path, game, [_region("Gamma", (0, 0, 0), (16, 16, 16))])
    assert "no such save directory" in message


def test_missing_regions_json_argument_exits_with_usage_error():
    with pytest.raises(SystemExit) as raised:
        subject.main([])
    assert raised.value.code == 2


# --------------------------------------------------------------------------- #
# Pairwise comparison
# --------------------------------------------------------------------------- #


def test_pairwise_intersections_and_coverage(tmp_path):
    game = _game(tmp_path)
    palette_a = [STONE, OAK_X]
    palette_b = [STONE, OAK_X, OAK_Z]
    _write_region(
        _region_path(game, "Alpha", 0, 0),
        {
            (0, 0): _chunk_bytes(
                [_section(0, palette_a, data=_pack(_indices_for([3000, 1096]), 2))], x=0, z=0),
            (1, 0): _chunk_bytes(
                [_section(0, palette_b, data=_pack(_indices_for([2500, 1000, 596]), 3))], x=1, z=0),
        },
    )
    document = _document(tmp_path, game, [
        _region("Alpha", (0, 0, 0), (16, 16, 16)),
        _region("Alpha", (16, 0, 0), (32, 16, 16)),
    ])
    pair = document["pairs"][0]
    assert pair["shared_state_count"] == 2
    assert pair["intersection_total"] == 3500
    assert pair["union_total"] == 4692
    assert pair["coverage_a"] == round(3500 / 4096, 6)
    assert pair["coverage_b"] == round(3500 / 4096, 6)
    assert pair["jaccard"] == round(3500 / 4692, 6)
    assert pair["shared_state_counts"] == {
        "minecraft:oak_log[axis=x]": [1096, 1000],
        "minecraft:stone": [3000, 2500],
    }
    assert document["totals"]["distinct_states"] == 3


def test_pair_with_no_non_air_blocks_reports_null_coverage(tmp_path):
    game = _game(tmp_path)
    _write_region(
        _region_path(game, "Alpha", 0, 0),
        {(0, 0): _chunk_bytes([_section(0, [], block_states=False)])},
    )
    document = _document(tmp_path, game, [
        _region("Alpha", (0, 0, 0), (16, 16, 16)),
        _region("Alpha", (16, 0, 0), (32, 16, 16), label="degenerate"),
    ])
    assert document["regions"][1]["requested_chunks"] == 1
    assert document["regions"][1]["missing_chunks"] == 1
    assert document["pairs"][0]["coverage_a"] is None


# --------------------------------------------------------------------------- #
# Determinism, seal and read-only behaviour
# --------------------------------------------------------------------------- #


def test_stdout_is_one_deterministic_sealed_json_document(tmp_path):
    game = _game(tmp_path)
    _write_region(
        _region_path(game, "Alpha", 0, 0),
        {(0, 0): _chunk_bytes([_section(0, [STONE, OAK_X], data=_pack(
            _indices_for([2048, 2048]), 2))])},
    )
    regions = [_region("Alpha", (0, 0, 0), (16, 16, 16))]
    request = _write_request(tmp_path, regions)
    first_code, first = _run(game, request)
    second_code, second = _run(game, request)
    assert first_code == second_code == 0
    assert first == second
    assert first.count("\n") == 1
    document = json.loads(first)
    assert document["status"] == "complete"
    seal = document.pop("seal")
    assert seal == subject.digest_json(document)


def test_success_and_refusal_write_nothing_under_the_game_directory(tmp_path):
    game = _game(tmp_path)
    _write_region(
        _region_path(game, "Alpha", 0, 0),
        {(0, 0): _chunk_bytes([_section(0, [STONE])])},
    )
    before = _snapshot(game)
    code, text = _scan(tmp_path, game, [_region("Alpha", (0, 0, 0), (16, 16, 16))])
    assert code == 0 and json.loads(text)["status"] == "complete"
    refusal_code, refusal_text = _scan(
        tmp_path, game, [_region("Alpha", (0, 0, 0), (16, 16, 16), bad="key")], name="bad.json")
    assert refusal_code == 2 and json.loads(refusal_text)["status"] == "refused"
    assert _snapshot(game) == before


def test_cli_subprocess_emits_json_only_on_stdout_and_writes_nothing(tmp_path):
    game = _game(tmp_path)
    _write_region(
        _region_path(game, "Alpha", 0, 0),
        {(0, 0): _chunk_bytes([_section(0, [STONE, OAK_X], data=_pack(
            _indices_for([1000, 3096]), 2))])},
    )
    request = _write_request(tmp_path, [_region("Alpha", (0, 0, 0), (16, 16, 16))])
    before = _snapshot(game)
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--game-dir", str(game),
         "--regions-json", str(request)],
        capture_output=True, text=True, check=False)
    assert completed.returncode == 0
    assert completed.stderr == ""
    assert completed.stdout.count("\n") == 1
    document = json.loads(completed.stdout)
    assert document["status"] == "complete"
    assert document["regions"][0]["non_air_total"] == 4096
    assert _snapshot(game) == before


def test_stdout_document_carries_the_documented_limits_and_formats(tmp_path):
    game = _game(tmp_path)
    _write_region(
        _region_path(game, "Alpha", 0, 0),
        {(0, 0): _chunk_bytes([_section(0, [STONE])])},
    )
    document = _document(tmp_path, game, [_region("Alpha", (0, 0, 0), (16, 16, 16))])
    assert document["limits"]["max_region_edge"] == 512
    assert document["limits"]["max_region_volume"] == 250000
    assert document["limits"]["max_compressed_chunk_bytes"] == 255 * SECTOR
    assert document["supported"]["compression"] == {"1": "gzip", "2": "deflate", "3": "none"}
    assert document["regions"][0]["data_versions"] == [3465]
