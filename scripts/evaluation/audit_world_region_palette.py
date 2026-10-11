"""Evaluation-only E7 read-only bounded Anvil world-state coverage scanner.

This module answers one question for the factory lead, and nothing else: *which
exact BlockStates exist inside these explicitly named, bounded block regions of
these already-present Minecraft saves?* It is a shortlisting aid for choosing
source worlds before any capture work starts. It is not a reconstruction input,
it never runs Minecraft, and it never opens a save for writing.

What it does, exactly:

* The only input is a strict JSON request naming one or more save directory
  names and one bounded integer block region per entry (``min`` inclusive,
  ``max`` exclusive). Save names must be single path components and must resolve
  directly under ``<game-dir>/saves``; traversal, absolute names and
  symlink/junction save, region or ``.mca`` paths are refusals.
* Every region is checked against the factory's own caps, imported from
  :mod:`mc_photogrammetry.dataset.schemas` (``MAX_REGION_EDGE=512``,
  ``MAX_REGION_VOLUME=250_000``), so the scanner cannot accept a region the
  capture factory would later reject.
* Only the ``.mca`` files and only the 32x32 chunk slots intersecting the
  requested regions are read. Unselected chunks, unselected region files, and
  all other saves are never opened. Nothing is staged or copied.
* Each present chunk's post-1.18 NBT is decoded (root-level ``sections`` with
  ``block_states`` palettes), normalized to ``Name`` plus sorted ``Properties``,
  and counted exactly, including singleton palettes (no ``data`` array) and
  packed palettes whose entries must not span two longs.
* The output is one strict JSON document on stdout: per-region exact state
  counts, non-air totals, missing chunks/sections, observed ``DataVersion``
  values, pairwise exact-state intersections with coverage ratios, and a
  deterministic SHA-256 seal. No output file is written and nothing else is
  printed to stdout.

What it deliberately refuses (fail closed, never a warning):

* the game directory, ``saves`` root, save directory, region directory or
  ``.mca`` file being link-like, any link-like *ancestor* of those paths, and
  any resolved save that does not sit directly under the resolved ``saves``
  root;
* external ``.mcc`` chunks (compression byte with ``0x80`` set), LZ4 and any
  other compression type outside gzip/deflate/none;
* pre-1.18 ``Level`` chunk layouts, ``DataVersion`` outside the documented
  range, chunks that are not ``minecraft:full``, and palettes with unsupported
  entry keys;
* half-zero location entries (exactly one of offset/sector count set),
  allocations that run past the file end, and any overlap between two occupied
  location-table allocations. All 1024 header slots are validated from the two
  header sectors alone, so an overlap involving an unselected chunk slot still
  refuses the scan without its payload ever being read;
* chunk frames that overflow their allocation, truncated/garbled compressed
  streams, decompressed chunks beyond the byte bound, truncated NBT, NBT beyond
  its depth/element budget, duplicate keys, invalid palette indexes, and missing
  required region files;
* NBT strings that are not strict Java modified UTF-8: malformed continuation
  bytes, overlong forms other than the ``C0 80`` NUL, raw four-byte UTF-8, and
  unpaired UTF-16 surrogates. Valid surrogate pairs are combined to the
  supplementary character they encode.

Exact state keys are delimiter-safe: the five structural characters ``\\ [ ] ,
=`` inside a block name, property key or property value are backslash-escaped,
so two genuinely different states can never collapse to one ``state_counts``
key. Ordinary names such as ``minecraft:oak_log[axis=x]`` are unchanged.

This tool is developer/evaluation scaffolding only. A palette count does not
prove a world is suitable, it accepts no end-user camera/bounds metadata, and it
does not feed the reconstruction program.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:  # direct-script execution without PYTHONPATH=src
    sys.path.insert(0, str(_SRC))

from mc_photogrammetry.dataset.hashing import (  # noqa: E402
    canonical_json,
    digest_json,
    is_link_like,
)
from mc_photogrammetry.dataset.schemas import (  # noqa: E402
    MAX_REGION_EDGE,
    MAX_REGION_VOLUME,
)

SCHEMA = "mc-photogrammetry-world-region-palette-scan-v1"
"""Schema identity of the scan payload."""

REQUEST_SCHEMA = "mc-photogrammetry-world-region-scan-request-v1"
"""Optional ``schema`` value the request may declare."""

ENGINE_VERSION = "e7-1r2"
"""Scanner revision recorded in the payload (E7R2 link-absolute + byte-bound repairs)."""

MODE = "read-only bounded Anvil region palette scan (evaluation-only)"

SECTOR = 4096
"""Anvil region files are addressed in 4096-byte sectors."""

HEADER_SECTORS = 2
"""The location and timestamp tables occupy the first two sectors."""

LOCATION_SLOTS = 32 * 32
"""One 4-byte location entry per 32x32 chunk slot in the header."""

SECTION_EDGE = 16
"""A chunk section is a 16-block cube (4096 positions)."""

SECTION_VOLUME = SECTION_EDGE ** 3

MIN_DATA_VERSION = 2865
"""1.18 release: the first layout with a root-level ``sections`` list."""

MAX_DATA_VERSION = 6000
"""Deliberate upper guard; versions above this have not been reviewed here."""

MAX_COMPRESSED_CHUNK_BYTES = 255 * SECTOR
"""Largest internal Anvil allocation (8-bit sector count). Bigger is external."""

MAX_DECOMPRESSED_CHUNK_BYTES = 64 * 1024 * 1024
"""Explicit decompression bomb guard for one chunk's NBT."""

MAX_NBT_DEPTH = 24
MAX_NBT_ELEMENTS = 2_000_000
MAX_REGIONS = 64

SUPPORTED_COMPRESSION = {1: "gzip", 2: "deflate", 3: "none"}
"""Documented region compression ids this scanner decodes. Anything else refuses."""

AIR_BLOCKS = frozenset({"minecraft:air", "minecraft:cave_air", "minecraft:void_air"})
"""Air-family states; they are counted, but reported apart from exact non-air states."""

TAG_END = 0
TAG_BYTE = 1
TAG_SHORT = 2
TAG_INT = 3
TAG_LONG = 4
TAG_FLOAT = 5
TAG_DOUBLE = 6
TAG_BYTE_ARRAY = 7
TAG_STRING = 8
TAG_LIST = 9
TAG_COMPOUND = 10
TAG_INT_ARRAY = 11
TAG_LONG_ARRAY = 12

_REQUEST_KEYS = frozenset({"schema", "regions"})
_REGION_KEYS = frozenset({"save", "min", "max", "label"})
_PALETTE_ENTRY_KEYS = frozenset({"Name", "Properties"})

_STATE_KEY_DELIMITERS = ("[", "]", ",", "=")
"""Structural characters that make a rendered exact-state key ambiguous."""


class ScanError(ValueError):
    """A typed refusal: the requested scan cannot be performed exactly."""


# --------------------------------------------------------------------------- #
# Java modified UTF-8 (the NBT string encoding)
# --------------------------------------------------------------------------- #


def _decode_modified_utf8(raw: bytes) -> str:
    """Decode canonical Java modified UTF-8, refusing malformed/noncanonical forms.

    NBT strings are Java *modified* UTF-8: NUL is the two-byte ``C0 80`` (a raw
    ``00`` byte is also accepted, matching ``readUTF``), BMP characters use one
    to three bytes, and characters above the BMP are encoded as a CESU-8 pair of
    three-byte surrogate code units. Anything malformed or noncanonical - a broken
    continuation byte, an overlong form other than the NUL form, a four-byte UTF-8
    sequence (modified UTF-8 has none) or an unpaired surrogate - is refused rather
    than silently replaced. This is stricter than Java's reader for malformed
    overlong encodings and lone surrogates; canonical strings emitted by Java remain
    compatible.
    """
    units: list[int] = []
    index = 0
    size = len(raw)
    while index < size:
        first = raw[index]
        if first < 0x80:
            units.append(first)
            index += 1
            continue
        if first >> 5 == 0b110:
            if index + 2 > size:
                raise ScanError("chunk NBT string ends inside a modified UTF-8 sequence")
            second = raw[index + 1]
            if second & 0xC0 != 0x80:
                raise ScanError("chunk NBT string has a malformed modified UTF-8 continuation")
            code = ((first & 0x1F) << 6) | (second & 0x3F)
            if code == 0:
                units.append(0)  # the canonical modified UTF-8 NUL, C0 80
            elif code < 0x80:
                raise ScanError("chunk NBT string has an overlong modified UTF-8 sequence")
            else:
                units.append(code)
            index += 2
            continue
        if first >> 4 == 0b1110:
            if index + 3 > size:
                raise ScanError("chunk NBT string ends inside a modified UTF-8 sequence")
            second = raw[index + 1]
            third = raw[index + 2]
            if second & 0xC0 != 0x80 or third & 0xC0 != 0x80:
                raise ScanError("chunk NBT string has a malformed modified UTF-8 continuation")
            code = ((first & 0x0F) << 12) | ((second & 0x3F) << 6) | (third & 0x3F)
            if code < 0x800:
                raise ScanError("chunk NBT string has an overlong modified UTF-8 sequence")
            units.append(code)  # surrogate code units are paired below
            index += 3
            continue
        raise ScanError("chunk NBT string is not valid Java modified UTF-8")

    text: list[str] = []
    position = 0
    while position < len(units):
        code = units[position]
        if 0xD800 <= code <= 0xDBFF:
            if position + 1 >= len(units) or not 0xDC00 <= units[position + 1] <= 0xDFFF:
                raise ScanError("chunk NBT string has an unpaired UTF-16 surrogate")
            low = units[position + 1]
            text.append(chr(0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)))
            position += 2
            continue
        if 0xDC00 <= code <= 0xDFFF:
            raise ScanError("chunk NBT string has an unpaired UTF-16 surrogate")
        text.append(chr(code))
        position += 1
    return "".join(text)


# --------------------------------------------------------------------------- #
# Minimal, bounded, big-endian NBT reader
# --------------------------------------------------------------------------- #


class _NbtArray(list):
    """A decoded NBT array that remembers its element width (1, 4 or 8 bytes)."""

    __slots__ = ("width",)

    def __init__(self, values: list[int], width: int) -> None:
        super().__init__(values)
        self.width = width


class _NbtReader:
    """Strict NBT reader with explicit depth, element and length bounds.

    Only the tags a modern chunk needs are decoded; every primitive is range
    checked against the remaining bytes, and the caller's element budget stops a
    pathological payload from turning one chunk into an unbounded allocation.
    """

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._position = 0
        self._budget = MAX_NBT_ELEMENTS

    @property
    def remaining(self) -> int:
        return len(self._data) - self._position

    def _take(self, size: int) -> bytes:
        if size < 0:
            raise ScanError("chunk NBT declares a negative length")
        end = self._position + size
        if end > len(self._data):
            raise ScanError("truncated chunk NBT")
        chunk = self._data[self._position:end]
        self._position = end
        return chunk

    def _spend(self, count: int = 1) -> None:
        self._budget -= count
        if self._budget < 0:
            raise ScanError("chunk NBT exceeds its element budget")

    def u1(self) -> int:
        return self._take(1)[0]

    def i1(self) -> int:
        value = self._take(1)[0]
        return value - 256 if value >= 128 else value

    def u2(self) -> int:
        return int.from_bytes(self._take(2), "big")

    def i2(self) -> int:
        value = self.u2()
        return value - 65536 if value >= 32768 else value

    def i4(self) -> int:
        return int.from_bytes(self._take(4), "big", signed=True)

    def i8(self) -> int:
        return int.from_bytes(self._take(8), "big", signed=True)

    def f4(self) -> float:
        return struct.unpack(">f", self._take(4))[0]

    def f8(self) -> float:
        return struct.unpack(">d", self._take(8))[0]

    def string(self) -> str:
        raw = self._take(self.u2())
        return _decode_modified_utf8(raw)

    def _array(self, width: int) -> _NbtArray:
        length = self.i4()
        if length < 0 or length > MAX_NBT_ELEMENTS:
            raise ScanError("chunk NBT array length is outside the element budget")
        self._spend(length)
        values = [int.from_bytes(self._take(width), "big", signed=True) for _ in range(length)]
        return _NbtArray(values, width)

    def payload(self, tag_type: int, depth: int) -> Any:
        self._spend()
        if depth > MAX_NBT_DEPTH:
            raise ScanError("chunk NBT nesting exceeds the depth bound")
        if tag_type == TAG_BYTE:
            return self.i1()
        if tag_type == TAG_SHORT:
            return self.i2()
        if tag_type == TAG_INT:
            return self.i4()
        if tag_type == TAG_LONG:
            return self.i8()
        if tag_type == TAG_FLOAT:
            return self.f4()
        if tag_type == TAG_DOUBLE:
            return self.f8()
        if tag_type == TAG_BYTE_ARRAY:
            return self._array(1)
        if tag_type == TAG_STRING:
            return self.string()
        if tag_type == TAG_LIST:
            element_type = self.u1()
            length = self.i4()
            if length < 0:
                raise ScanError("chunk NBT declares a negative list length")
            if element_type == TAG_END and length:
                raise ScanError("chunk NBT list of TAG_End cannot hold entries")
            self._spend(length)
            return [self.payload(element_type, depth + 1) for _ in range(length)]
        if tag_type == TAG_COMPOUND:
            value: dict[str, Any] = {}
            while True:
                child_type = self.u1()
                if child_type == TAG_END:
                    return value
                key = self.string()
                if key in value:
                    raise ScanError("chunk NBT compound repeats a key")
                value[key] = self.payload(child_type, depth + 1)
        if tag_type == TAG_INT_ARRAY:
            return self._array(4)
        if tag_type == TAG_LONG_ARRAY:
            return self._array(8)
        raise ScanError(f"chunk NBT uses unsupported tag type {tag_type}")


def _parse_chunk_nbt(data: bytes) -> dict[str, Any]:
    """Parse one decompressed chunk payload into a root compound."""
    if len(data) > MAX_DECOMPRESSED_CHUNK_BYTES:
        raise ScanError("decompressed chunk exceeds the byte bound")
    reader = _NbtReader(data)
    if reader.u1() != TAG_COMPOUND:
        raise ScanError("chunk NBT root is not a compound")
    reader.string()  # root name; Minecraft writes an empty name
    root = reader.payload(TAG_COMPOUND, 1)
    if reader.remaining != 0:
        raise ScanError("chunk NBT has trailing bytes after its root compound")
    return root


# --------------------------------------------------------------------------- #
# Request parsing and path safety
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RegionRequest:
    """One validated, bounded block region inside one named save."""

    index: int
    save: str
    minimum: tuple[int, int, int]
    maximum: tuple[int, int, int]
    label: str | None


def default_game_dir() -> Path:
    """The platform's usual ``.minecraft`` location (never created here)."""
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / ".minecraft"
    return Path.home() / ".minecraft"


def _safe_save_name(value: Any) -> str:
    """Validate one save directory name before it is joined to a path."""
    if not isinstance(value, str) or not value or value in (".", ".."):
        raise ScanError(f"unsafe save directory name: {value!r}")
    if any(character in value for character in ("/", "\\", "\0", ":")):
        raise ScanError(f"save directory name must be one path component: {value!r}")
    if value != value.strip() or value.endswith("."):
        raise ScanError(f"unsafe save directory name: {value!r}")
    if Path(value).is_absolute():
        raise ScanError(f"save directory name must be relative: {value!r}")
    return value


def _triple(value: Any, label: str) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise ScanError(f"region {label} must be a list of three integers")
    axis: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ScanError(f"region {label} must contain only integers")
        axis.append(int(item))
    return (axis[0], axis[1], axis[2])


def _validate_region(index: int, item: Any) -> RegionRequest:
    if not isinstance(item, dict):
        raise ScanError(f"region {index} must be a JSON object")
    unknown = sorted(set(item) - _REGION_KEYS)
    if unknown:
        raise ScanError(f"region {index} has unsupported key(s): {unknown}")
    save = _safe_save_name(item.get("save"))
    minimum = _triple(item.get("min"), "min")
    maximum = _triple(item.get("max"), "max")
    label = item.get("label")
    if label is not None and (not isinstance(label, str) or not label):
        raise ScanError(f"region {index} label must be a non-empty string")
    volume = 1
    for axis in range(3):
        edge = maximum[axis] - minimum[axis]
        if edge <= 0:
            raise ScanError(f"region {index} max must exceed min on every axis")
        if edge > MAX_REGION_EDGE:
            raise ScanError(
                f"region {index} edge {edge} exceeds MAX_REGION_EDGE={MAX_REGION_EDGE}")
        volume *= edge
    if volume > MAX_REGION_VOLUME:
        raise ScanError(
            f"region {index} volume {volume} exceeds MAX_REGION_VOLUME={MAX_REGION_VOLUME}")
    return RegionRequest(index=index, save=save, minimum=minimum, maximum=maximum, label=label)


def load_request(path: Path | str) -> tuple[RegionRequest, ...]:
    """Read and validate the strict ``--regions-json`` request."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as error:
        raise ScanError(f"cannot read regions JSON {path}: {error}") from error
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ScanError(f"regions JSON is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ScanError("regions JSON must be an object")
    unknown = sorted(set(payload) - _REQUEST_KEYS)
    if unknown:
        raise ScanError(f"regions JSON has unsupported key(s): {unknown}")
    if "schema" in payload and payload["schema"] != REQUEST_SCHEMA:
        raise ScanError(f"regions JSON schema must be {REQUEST_SCHEMA!r}")
    regions = payload.get("regions")
    if not isinstance(regions, list) or not regions:
        raise ScanError("regions JSON must hold a non-empty 'regions' list")
    if len(regions) > MAX_REGIONS:
        raise ScanError(f"regions JSON requests {len(regions)} regions, above the cap {MAX_REGIONS}")
    return tuple(_validate_region(index, item) for index, item in enumerate(regions))


def _assert_link_free(path: Path, label: str) -> Path:
    """Refuse a path whose leaf or any ancestor is a symlink/reparse point.

    Checking only the leaf would let a linked *intermediate* directory redirect
    the read outside the directory the caller believes it is bounding. The path
    is first made absolute *lexically* (``os.path.abspath``: it joins the cwd and
    collapses ``.``/``..`` but never resolves a symlink), because the ``parents``
    of a relative path stop at ``.`` and would otherwise leave every directory
    above the cwd unchecked. The lexical absolute form is returned so every
    caller keeps guarding the same path the ancestors were validated against.
    """
    absolute = Path(os.path.abspath(path))
    for component in (absolute, *absolute.parents):
        if is_link_like(component):
            raise ScanError(f"{label} is link-like and is refused: {component}")
    return absolute


def _resolve_save(saves_root: Path, resolved_saves_root: Path, name: str) -> Path:
    """Resolve one save directory directly under ``<game-dir>/saves``."""
    save_root = _assert_link_free(saves_root / name, "save path")
    if not save_root.is_dir():
        raise ScanError(f"no such save directory: {save_root}")
    resolved = save_root.resolve()
    if resolved != resolved_saves_root and resolved_saves_root not in resolved.parents:
        raise ScanError(f"save path escapes <game-dir>/saves: {save_root}")
    return save_root


# --------------------------------------------------------------------------- #
# Chunk decoding
# --------------------------------------------------------------------------- #


def _decompress(payload: bytes, kind: int) -> bytes:
    """Decode one chunk frame under the explicit decompressed byte bound."""
    if kind == 3:
        if len(payload) > MAX_DECOMPRESSED_CHUNK_BYTES:
            raise ScanError("chunk exceeds the decompressed byte bound")
        return payload
    window = 31 if kind == 1 else 15  # gzip / zlib
    engine = zlib.decompressobj(window)
    try:
        data = engine.decompress(payload, MAX_DECOMPRESSED_CHUNK_BYTES)
    except zlib.error as error:
        raise ScanError(f"chunk decompression failed: {error}") from error
    if engine.unconsumed_tail:
        raise ScanError("chunk exceeds the decompressed byte bound")
    if not engine.eof:
        raise ScanError("chunk compressed stream is truncated")
    if engine.unused_data:
        raise ScanError("chunk compressed stream has trailing bytes")
    return data


def _normalized_state(name: str, properties: Any) -> tuple[str, tuple[tuple[str, str], ...]]:
    if not isinstance(name, str) or not name:
        raise ScanError("section palette entry has no non-empty 'Name'")
    normalized = name if ":" in name else f"minecraft:{name}"
    if properties is None:
        return normalized, ()
    if not isinstance(properties, dict):
        raise ScanError("section palette entry 'Properties' must be a compound")
    pairs: list[tuple[str, str]] = []
    for key, value in properties.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ScanError("section palette properties must map strings to strings")
        pairs.append((key, value))
    return normalized, tuple(sorted(pairs))


def _escape_component(value: str) -> str:
    """Escape the structural delimiters of an exact-state key component.

    The escape is applied to component text only; the ``[``, ``,``, ``]`` and
    ``=`` the renderer adds are left literal, so the rendered key stays
    one-to-one with the ``(name, sorted properties)`` tuple.
    """
    escaped = value.replace("\\", "\\\\")
    for delimiter in _STATE_KEY_DELIMITERS:
        escaped = escaped.replace(delimiter, "\\" + delimiter)
    return escaped


def _state_key(state: tuple[str, tuple[tuple[str, str], ...]]) -> str:
    name, properties = state
    if not properties:
        return _escape_component(name)
    rendered = ",".join(
        f"{_escape_component(key)}={_escape_component(value)}" for key, value in properties)
    return f"{_escape_component(name)}[{rendered}]"


def _parse_palette(block_states: dict[str, Any]) -> list[tuple[str, tuple[tuple[str, str], ...]]]:
    palette = block_states.get("palette")
    if not isinstance(palette, list) or not palette:
        raise ScanError("section block_states has no non-empty palette")
    states: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    for entry in palette:
        if not isinstance(entry, dict):
            raise ScanError("section palette entry is not a compound")
        unknown = sorted(set(entry) - _PALETTE_ENTRY_KEYS)
        if unknown:
            raise ScanError(f"section palette entry has unsupported key(s): {unknown}")
        states.append(_normalized_state(entry.get("Name"), entry.get("Properties")))
    return states


def _section_indices(data: Any, palette_size: int) -> list[int]:
    """Decode one section's 4096 palette indexes (singleton or packed)."""
    if data is None:
        if palette_size != 1:
            raise ScanError("section has a multi-entry palette but no data array")
        return [0] * SECTION_VOLUME
    if not isinstance(data, _NbtArray) or data.width != 8:
        raise ScanError("section 'data' is not a long array")
    bits = max(4, (palette_size - 1).bit_length())
    per_long = 64 // bits
    required = -(-SECTION_VOLUME // per_long)
    if len(data) != required:
        raise ScanError(
            f"section data array holds {len(data)} long(s), expected {required}")
    mask = (1 << bits) - 1
    indices: list[int] = []
    for value in data:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ScanError("section 'data' must contain only longs")
        unsigned = value & ((1 << 64) - 1)
        for slot in range(per_long):
            indices.append((unsigned >> (slot * bits)) & mask)
    indices = indices[:SECTION_VOLUME]
    for index in indices:
        if index >= palette_size:
            raise ScanError("section palette index is out of range")
    return indices


@dataclass(frozen=True, slots=True)
class ChunkStats:
    """What one present chunk contributed to one region."""

    data_version: int
    counts: Counter
    air_total: int
    scanned_positions: int
    sections_requested: int
    sections_present: int

    @property
    def missing_sections(self) -> int:
        return self.sections_requested - self.sections_present


def _overlap(first_min: int, first_max: int, second_min: int, second_max: int) -> int:
    """Length of the intersection of two half-open integer intervals."""
    return max(0, min(first_max, second_max) - max(first_min, second_min))


def _count_chunk(
    root: dict[str, Any],
    chunk_x: int,
    chunk_z: int,
    minimum: tuple[int, int, int],
    maximum: tuple[int, int, int],
) -> ChunkStats:
    """Validate one chunk compound and count its in-region block states."""
    data_version = root.get("DataVersion")
    if not isinstance(data_version, int) or isinstance(data_version, bool):
        raise ScanError("chunk NBT has no integer DataVersion")
    if not MIN_DATA_VERSION <= data_version <= MAX_DATA_VERSION:
        raise ScanError(
            f"chunk DataVersion {data_version} is outside "
            f"[{MIN_DATA_VERSION}, {MAX_DATA_VERSION}]")
    if "Level" in root:
        raise ScanError("pre-1.18 'Level' chunk layout is not supported")
    for key, expected, label in (("xPos", chunk_x, "x"), ("zPos", chunk_z, "z")):
        if key not in root:
            continue
        position = root[key]
        if not isinstance(position, int) or isinstance(position, bool):
            raise ScanError(f"chunk {key} is not an integer")
        if int(position) != expected:
            raise ScanError(f"chunk {key} does not match its {label} region slot")
    if root.get("Status") != "minecraft:full":
        raise ScanError(f"chunk is not fully generated (Status={root.get('Status')!r})")
    sections = root.get("sections")
    if not isinstance(sections, list):
        raise ScanError("chunk NBT has no 'sections' list")

    minimum_x, minimum_y, minimum_z = minimum
    maximum_x, maximum_y, maximum_z = maximum
    first_section = minimum_y // SECTION_EDGE
    last_section = (maximum_y - 1) // SECTION_EDGE
    requested_sections = last_section - first_section + 1

    counts: Counter = Counter()
    air_total = 0
    scanned_positions = 0
    present: set[int] = set()
    for section in sections:
        if not isinstance(section, dict):
            raise ScanError("chunk section is not a compound")
        section_y = section.get("Y")
        if not isinstance(section_y, int) or isinstance(section_y, bool):
            raise ScanError("chunk section has no integer Y")
        if section_y in present:
            raise ScanError(f"chunk repeats section Y={section_y}")
        present.add(section_y)
        base_y = section_y * SECTION_EDGE
        if base_y + SECTION_EDGE <= minimum_y or base_y >= maximum_y:
            continue
        block_states = section.get("block_states")
        if block_states is None:
            # A section without block_states is entirely air; count the
            # in-region positions so the air total stays honest.
            overlap = (
                _overlap(chunk_x * SECTION_EDGE, chunk_x * SECTION_EDGE + SECTION_EDGE,
                         minimum_x, maximum_x)
                * _overlap(base_y, base_y + SECTION_EDGE, minimum_y, maximum_y)
                * _overlap(chunk_z * SECTION_EDGE, chunk_z * SECTION_EDGE + SECTION_EDGE,
                           minimum_z, maximum_z)
            )
            air_total += overlap
            scanned_positions += overlap
            continue
        if not isinstance(block_states, dict):
            raise ScanError("chunk section 'block_states' is not a compound")
        palette = _parse_palette(block_states)
        indices = _section_indices(block_states.get("data"), len(palette))
        for position, index in enumerate(indices):
            local_y, rest = divmod(position, SECTION_EDGE * SECTION_EDGE)
            local_z, local_x = divmod(rest, SECTION_EDGE)
            world_x = chunk_x * SECTION_EDGE + local_x
            world_y = base_y + local_y
            world_z = chunk_z * SECTION_EDGE + local_z
            if not (minimum_x <= world_x < maximum_x):
                continue
            if not (minimum_z <= world_z < maximum_z):
                continue
            if not (minimum_y <= world_y < maximum_y):
                continue
            scanned_positions += 1
            state = palette[index]
            if state[0] in AIR_BLOCKS:
                air_total += 1
                continue
            counts[_state_key(state)] += 1

    in_window = {y for y in present if first_section <= y <= last_section}
    return ChunkStats(
        data_version=int(data_version),
        counts=counts,
        air_total=air_total,
        scanned_positions=scanned_positions,
        sections_requested=requested_sections,
        sections_present=len(in_window),
    )


# --------------------------------------------------------------------------- #
# Region scanning
# --------------------------------------------------------------------------- #


def _chunk_columns(minimum: tuple[int, int, int], maximum: tuple[int, int, int]):
    """The inclusive chunk-coordinate span intersecting a bounded region."""
    first_x = minimum[0] >> 4
    last_x = (maximum[0] - 1) >> 4
    first_z = minimum[2] >> 4
    last_z = (maximum[2] - 1) >> 4
    for chunk_x in range(first_x, last_x + 1):
        for chunk_z in range(first_z, last_z + 1):
            yield chunk_x, chunk_z


def _location_table(
    header: bytes, size: int, region_path: Path
) -> list[tuple[int, int] | None]:
    """Validate all 1024 location entries from the two header sectors alone.

    Returns ``(start, sector_count)`` for every occupied slot and ``None`` for a
    genuinely missing chunk (both fields zero). A half-zero entry is malformed,
    every occupied allocation must start after the header and fit inside the
    file, and no two occupied allocations may overlap. Because this runs before
    any seek, an overlap involving an *unselected* chunk slot refuses the scan
    without that chunk's payload ever being read.
    """
    entries: list[tuple[int, int] | None] = []
    allocations: list[tuple[int, int, int]] = []
    for slot in range(LOCATION_SLOTS):
        base = slot * 4
        offset = int.from_bytes(header[base:base + 3], "big")
        sector_count = header[base + 3]
        if offset == 0 and sector_count == 0:
            entries.append(None)
            continue
        if offset == 0 or sector_count == 0:
            raise ScanError(
                f"region location slot {slot} is half-zero "
                f"(offset={offset}, sectors={sector_count}) in {region_path.name}")
        if offset < HEADER_SECTORS:
            raise ScanError("chunk sector offset points into the region header")
        start = offset * SECTOR
        end = start + sector_count * SECTOR
        if end > size:
            raise ScanError(
                f"chunk sector allocation extends beyond the region file: {region_path.name}")
        entries.append((start, sector_count))
        allocations.append((start, end, slot))
    allocations.sort()
    previous_end = HEADER_SECTORS * SECTOR
    for start, end, slot in allocations:
        if start < previous_end:
            raise ScanError(
                f"region chunk allocations overlap at slot {slot}: {region_path.name}")
        previous_end = end
    return entries


def _scan_region(saves_root: Path, resolved_saves_root: Path, request: RegionRequest) -> dict[str, Any]:
    save_root = _resolve_save(saves_root, resolved_saves_root, request.save)
    region_dir = _assert_link_free(save_root / "region", "region directory")
    if not region_dir.is_dir():
        raise ScanError(f"save {request.save!r} has no region directory: {region_dir}")

    by_region: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for chunk_x, chunk_z in _chunk_columns(request.minimum, request.maximum):
        by_region.setdefault((chunk_x >> 5, chunk_z >> 5), []).append((chunk_x, chunk_z))

    counts: Counter = Counter()
    air_total = 0
    scanned_positions = 0
    scanned_chunks = 0
    missing_chunks = 0
    sections_requested = 0
    sections_present = 0
    data_versions: set[int] = set()

    for (region_x, region_z), coordinates in sorted(by_region.items()):
        region_path = _assert_link_free(
            region_dir / f"r.{region_x}.{region_z}.mca", "region file")
        if not region_path.is_file():
            raise ScanError(f"missing required region file: {region_path}")
        size = region_path.stat().st_size
        if size < HEADER_SECTORS * SECTOR:
            raise ScanError(f"region file is smaller than its two-sector header: {region_path}")
        with region_path.open("rb") as handle:
            header = handle.read(HEADER_SECTORS * SECTOR)
            if len(header) != HEADER_SECTORS * SECTOR:
                raise ScanError(f"region file header is truncated: {region_path}")
            locations = _location_table(header, size, region_path)
            for chunk_x, chunk_z in sorted(coordinates):
                slot = (chunk_x & 31) + (chunk_z & 31) * 32
                entry = locations[slot]
                if entry is None:
                    missing_chunks += 1
                    continue
                start, sector_count = entry
                handle.seek(start)
                frame = handle.read(5)
                if len(frame) != 5:
                    raise ScanError("chunk frame header is truncated")
                declared = int.from_bytes(frame[:4], "big")
                if declared < 1:
                    raise ScanError("chunk frame declares an empty payload")
                payload_length = declared - 1
                if payload_length > MAX_COMPRESSED_CHUNK_BYTES:
                    raise ScanError("compressed chunk frame exceeds the byte bound")
                if 5 + payload_length > sector_count * SECTOR:
                    raise ScanError("chunk frame overflows its allocated sectors")
                if start + 5 + payload_length > size:
                    raise ScanError("chunk frame is truncated by the region file end")
                version = frame[4]
                if version & 0x80:
                    raise ScanError(
                        "external .mcc chunk is not supported "
                        f"(expected c.{chunk_x}.{chunk_z}.mcc next to {region_path.name})")
                kind = version & 0x7F
                if kind not in SUPPORTED_COMPRESSION:
                    raise ScanError(f"unsupported chunk compression type {kind}")
                payload = handle.read(payload_length)
                if len(payload) != payload_length:
                    raise ScanError("chunk frame payload is truncated")

                root = _parse_chunk_nbt(_decompress(payload, kind))
                stats = _count_chunk(root, chunk_x, chunk_z, request.minimum, request.maximum)
                data_versions.add(stats.data_version)
                counts.update(stats.counts)
                air_total += stats.air_total
                scanned_positions += stats.scanned_positions
                sections_requested += stats.sections_requested
                sections_present += stats.sections_present
                scanned_chunks += 1

    non_air_total = sum(counts.values())
    return {
        "index": request.index,
        "label": request.label,
        "save": request.save,
        "min": list(request.minimum),
        "max": list(request.maximum),
        "requested_chunks": sum(len(value) for value in by_region.values()),
        "scanned_chunks": scanned_chunks,
        "missing_chunks": missing_chunks,
        "sections_requested": sections_requested,
        "sections_present": sections_present,
        "missing_sections": sections_requested - sections_present,
        "data_versions": sorted(data_versions),
        "scanned_block_positions": scanned_positions,
        "air_total": air_total,
        "non_air_total": non_air_total,
        "state_counts": {key: counts[key] for key in sorted(counts)},
    }


def _coverage(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 6)


def _pairwise(regions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(regions):
        left_counts = left["state_counts"]
        for right_index in range(left_index + 1, len(regions)):
            right = regions[right_index]
            right_counts = right["state_counts"]
            shared = sorted(set(left_counts) & set(right_counts))
            intersection = sum(min(left_counts[key], right_counts[key]) for key in shared)
            union = sum(
                max(left_counts.get(key, 0), right_counts.get(key, 0))
                for key in set(left_counts) | set(right_counts)
            )
            pairs.append({
                "a": left_index,
                "b": right_index,
                "shared_state_count": len(shared),
                "intersection_total": intersection,
                "union_total": union,
                "jaccard": _coverage(intersection, union),
                "coverage_a": _coverage(intersection, left["non_air_total"]),
                "coverage_b": _coverage(intersection, right["non_air_total"]),
                "shared_state_counts": {
                    key: [left_counts[key], right_counts[key]] for key in shared
                },
            })
    return pairs


# --------------------------------------------------------------------------- #
# Report assembly and CLI
# --------------------------------------------------------------------------- #


def _supported_formats() -> dict[str, Any]:
    return {
        "dimension": "<game-dir>/saves/<save>/region/r.<x>.<z>.mca (overworld) only",
        "chunk_layout": "post-1.18 root-level 'sections' with per-section 'block_states' palettes",
        "compression": {str(key): value for key, value in sorted(SUPPORTED_COMPRESSION.items())},
        "air_states": sorted(AIR_BLOCKS),
        "refused": [
            "external .mcc chunks (compression byte bit 0x80)",
            "LZ4 and any compression type other than 1, 2 and 3",
            "pre-1.18 'Level' chunk layout",
            f"DataVersion outside [{MIN_DATA_VERSION}, {MAX_DATA_VERSION}]",
            "chunk Status other than minecraft:full",
            "palette entries with keys other than Name and Properties",
            "link-like game/saves/save/region/.mca paths or intermediates",
            "half-zero location entries, allocations past EOF and overlapping allocations",
            "NBT strings that are not strict Java modified UTF-8",
            "malformed sectors, invalid palette indexes and truncated data",
        ],
    }


def _limits() -> dict[str, int]:
    return {
        "max_region_edge": MAX_REGION_EDGE,
        "max_region_volume": MAX_REGION_VOLUME,
        "max_regions": MAX_REGIONS,
        "max_compressed_chunk_bytes": MAX_COMPRESSED_CHUNK_BYTES,
        "max_decompressed_chunk_bytes": MAX_DECOMPRESSED_CHUNK_BYTES,
        "min_data_version": MIN_DATA_VERSION,
        "max_data_version": MAX_DATA_VERSION,
        "max_nbt_depth": MAX_NBT_DEPTH,
        "max_nbt_elements": MAX_NBT_ELEMENTS,
    }


def build_report(game_dir: Path | str, regions_json: Path | str) -> dict[str, Any]:
    """Perform the requested read-only scan and return the sealed payload."""
    requests = load_request(regions_json)
    game_root = _assert_link_free(Path(game_dir).expanduser(), "game directory")
    if not game_root.is_dir():
        raise ScanError(f"no such game directory: {game_root}")
    saves_root = _assert_link_free(game_root / "saves", "saves directory")
    if not saves_root.is_dir():
        raise ScanError(f"no saves directory under the game directory: {saves_root}")
    resolved_game_root = game_root.resolve()
    resolved_saves_root = saves_root.resolve()
    if resolved_saves_root.parent != resolved_game_root:
        raise ScanError(f"saves directory is not directly under the game directory: {saves_root}")

    regions = [
        _scan_region(saves_root, resolved_saves_root, request) for request in requests
    ]
    pairs = _pairwise(regions)
    distinct: set[str] = set()
    for region in regions:
        distinct.update(region["state_counts"])
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "engine_version": ENGINE_VERSION,
        "mode": MODE,
        "status": "complete",
        "game_dir": str(game_root),
        "limits": _limits(),
        "supported": _supported_formats(),
        "regions": regions,
        "pairs": pairs,
        "totals": {
            "regions": len(regions),
            "requested_chunks": sum(region["requested_chunks"] for region in regions),
            "scanned_chunks": sum(region["scanned_chunks"] for region in regions),
            "missing_chunks": sum(region["missing_chunks"] for region in regions),
            "missing_sections": sum(region["missing_sections"] for region in regions),
            "scanned_block_positions": sum(
                region["scanned_block_positions"] for region in regions),
            "non_air_total": sum(region["non_air_total"] for region in regions),
            "distinct_states": len(distinct),
        },
    }
    document["seal"] = digest_json(document)
    return document


def _refusal_document(kind: str, message: str) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "engine_version": ENGINE_VERSION,
        "mode": MODE,
        "status": "refused",
        "error": {"kind": kind, "message": message},
    }
    document["seal"] = digest_json(document)
    return document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audit_world_region_palette.py",
        description=(
            "Read-only bounded scan of exact BlockStates in selected regions of "
            "existing Minecraft saves (evaluation-only)."),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--game-dir",
        default=str(default_game_dir()),
        help="game directory holding 'saves'; defaults to the usual .minecraft",
    )
    parser.add_argument(
        "--regions-json",
        required=True,
        help="strict JSON request naming saves and bounded block regions",
    )
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    """Run the scanner. Exactly one JSON document reaches stdout."""
    parser = build_parser()
    args = parser.parse_args(argv)
    stream = sys.stdout if stdout is None else stdout
    try:
        document = build_report(args.game_dir, args.regions_json)
    except ScanError as error:
        print(canonical_json(_refusal_document("refused", str(error))), file=stream)
        return 2
    except OSError as error:
        print(canonical_json(_refusal_document("io_error", str(error))), file=stream)
        return 2
    print(canonical_json(document), file=stream)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
