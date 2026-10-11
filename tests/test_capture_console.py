"""Focused tests for encoding-safe dataset inventory output (DS-26).

The defect: ``prepare_capture_dataset.py --list`` aborted with
``UnicodeEncodeError: 'charmap' codec can't encode characters`` on a strict cp1252
stdout as soon as one save carried a non-cp1252 ``level.dat`` ``LevelName``, so
every save after it vanished from the listing.

These tests pin the repair: the listing completes on a strict legacy stream, only
the unencodable characters are escaped, representable Unicode is preserved
verbatim, UTF-8 consoles still see the real names, output is deterministic, and
no inventory record is altered or skipped by rendering.
"""
from __future__ import annotations

import importlib.util
import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mc_photogrammetry.dataset import (
    escape_for_encoding,
    inventory_saves,
    is_encodable,
    stream_encoding,
    write_line,
    write_safe,
)

ROOT = Path(__file__).resolve().parents[1]

#: The shape of the real failure: a 12-character CJK world name that cp1252
#: cannot represent at all.
CJK_LEVEL_NAME = "\u9ad8\u538b\u9ad8\u706b\u529b\u4e0b\u5f39\u4e09\u5411\u8f70\u70b8\u673a"
CJK_SAVE_NAME = "\u9ad8\u538b\u5b58\u6863"
CJK_ESCAPED = ("\\u9ad8\\u538b\\u9ad8\\u706b\\u529b\\u4e0b\\u5f39\\u4e09\\u5411"
               "\\u8f70\\u70b8\\u673a")
CJK_SAVE_ESCAPED = "\\u9ad8\\u538b\\u5b58\\u6863"


def _legacy_level_dat(path: Path, *, level_name: str) -> None:
    import nbtlib
    from nbtlib import tag

    path.parent.mkdir(parents=True, exist_ok=True)
    nbtlib.File(tag.Compound({"Data": tag.Compound({
        "LevelName": tag.String(level_name),
        "LastPlayed": tag.Long(1_700_000_000_000),
        "SpawnX": tag.Int(3),
        "SpawnY": tag.Int(70),
        "SpawnZ": tag.Int(-4),
    })}), gzipped=True).save(path)


def _game_with_saves(tmp_path: Path) -> Path:
    """Four saves: ASCII, non-cp1252 level name, cp1252 accent, non-cp1252 dir."""
    game = tmp_path / "game"
    for directory, level_name in (
        ("Alpha", "Alpha World"),
        ("Beta", CJK_LEVEL_NAME),
        ("Gamma", "Caf\u00e9 World"),
        (CJK_SAVE_NAME, "High Voltage"),
    ):
        world = game / "saves" / directory
        _legacy_level_dat(world / "level.dat", level_name=level_name)
        region = world / "region"
        region.mkdir(parents=True, exist_ok=True)
        (region / "r.0.0.mca").write_bytes(b"region")
    return game


def _strict_text_stream(encoding: str):
    """A real, strict byte stream: writing an unencodable character must raise."""
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding=encoding, errors="strict", newline="")
    return stream, buffer


def _prepare_module():
    path = ROOT / "scripts" / "prepare_capture_dataset.py"
    spec = importlib.util.spec_from_file_location("ds26_prepare_capture_dataset", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _render(saves, encoding: str) -> str:
    stream, buffer = _strict_text_stream(encoding)
    assert _prepare_module()._print_saves(saves, stream=stream) == 0
    stream.flush()
    return buffer.getvalue().decode(encoding)


# --- the reported crash ------------------------------------------------------


def test_strict_cp1252_listing_completes_and_keeps_every_record(tmp_path):
    game = _game_with_saves(tmp_path)
    saves = inventory_saves(game)
    assert len(saves) == 4

    rendered = _render(saves, "cp1252")
    lines = rendered.splitlines()
    # One header plus one line per save: nothing was dropped to dodge the error.
    assert len(lines) == len(saves) + 1
    assert lines[0].startswith("name")

    for save in saves:
        assert save.name[:40] in rendered or escape_for_encoding(save.name, "cp1252") in rendered

    # The unencodable characters are escaped, deterministically and readably.
    assert CJK_ESCAPED in rendered
    assert CJK_SAVE_ESCAPED in rendered
    # A cp1252-representable non-ASCII name is preserved, not escaped.
    assert "Caf\u00e9 World" in rendered
    assert "\\u00e9" not in rendered


def test_strict_cp1252_listing_preserves_inventory_order(tmp_path):
    saves = inventory_saves(_game_with_saves(tmp_path))
    rendered = _render(saves, "cp1252")
    data_lines = rendered.splitlines()[1:]
    # The name column is the first, 40-wide field; escaping can widen it, so read
    # the field back the same way the writer truncates it.
    rendered_names = [line[:40].rstrip() for line in data_lines]
    expected = [escape_for_encoding(save.name[:40], "cp1252") for save in saves]
    assert rendered_names == expected


def test_rendering_does_not_change_or_skip_any_save_record(tmp_path):
    game = _game_with_saves(tmp_path)
    before = [
        (save.name, save.level_name, save.path, save.size_bytes, save.modified_utc)
        for save in inventory_saves(game)
    ]
    files_before = sorted(str(path.relative_to(game)) for path in game.rglob("*"))

    _render(inventory_saves(game), "cp1252")

    after = [
        (save.name, save.level_name, save.path, save.size_bytes, save.modified_utc)
        for save in inventory_saves(game)
    ]
    assert after == before
    # Reading and rendering never touch the source tree.
    assert sorted(str(path.relative_to(game)) for path in game.rglob("*")) == files_before
    assert before[1][1] == CJK_LEVEL_NAME


# --- preservation where the console supports it ------------------------------


def test_utf8_stream_preserves_unicode_verbatim(tmp_path):
    saves = inventory_saves(_game_with_saves(tmp_path))
    rendered = _render(saves, "utf-8")
    lines = rendered.splitlines()
    assert len(lines) == len(saves) + 1
    for save in saves:
        assert save.name[:40] in rendered
    assert CJK_LEVEL_NAME in rendered
    assert CJK_SAVE_NAME in rendered
    assert "Caf\u00e9 World" in rendered
    assert "\\u9ad8" not in rendered


def test_utf8_and_cp1252_renderings_agree_after_escaping(tmp_path):
    saves = inventory_saves(_game_with_saves(tmp_path))
    utf8_text = _render(saves, "utf-8")
    cp1252_text = _render(saves, "cp1252")
    # The legacy stream shows exactly the UTF-8 listing with the unencodable
    # characters escaped - same records, same order, same columns.
    assert escape_for_encoding(utf8_text, "cp1252") == cp1252_text


# --- determinism -------------------------------------------------------------


def test_listing_is_byte_deterministic_on_both_encodings(tmp_path):
    saves = inventory_saves(_game_with_saves(tmp_path))
    for encoding in ("cp1252", "utf-8"):
        first = _render(saves, encoding)
        second = _render(saves, encoding)
        assert first == second
        assert first.encode(encoding) == second.encode(encoding)


def test_empty_inventory_is_still_writable_on_a_legacy_stream(tmp_path):
    game = tmp_path / "game"
    (game / "saves").mkdir(parents=True)
    stream, buffer = _strict_text_stream("ascii")
    assert _prepare_module()._print_saves(inventory_saves(game), stream=stream) == 0
    stream.flush()
    assert buffer.getvalue() == b"no saves found\n"


# --- the helper contract -----------------------------------------------------


def test_escape_for_encoding_escapes_only_unencodable_characters():
    assert escape_for_encoding("Caf\u00e9", "cp1252") == "Caf\u00e9"
    assert escape_for_encoding("\u9ad8\u538b", "cp1252") == "\\u9ad8\\u538b"
    assert escape_for_encoding("a\u9ad8b", "cp1252") == "a\\u9ad8b"
    # A lone surrogate (as Python can see in an exotic filesystem name) escapes
    # instead of killing the stream, and a codec-less stream needs no escaping.
    assert escape_for_encoding("\ud800", "cp1252") == "\\ud800"
    assert escape_for_encoding("\u9ad8", None) == "\u9ad8"
    assert escape_for_encoding("\u9ad8", "utf-8") == "\u9ad8"


def test_encodability_queries_match_the_codec():
    assert is_encodable("Caf\u00e9", "cp1252") is True
    assert is_encodable("\u9ad8", "cp1252") is False
    assert is_encodable("\u9ad8", "utf-8") is True
    assert is_encodable("\u9ad8", None) is True
    # An unknown codec cannot be judged here; the stream stays the authority.
    assert is_encodable("\u9ad8", "definitely-not-a-codec") is True
    assert stream_encoding(io.StringIO()) is None


def test_write_safe_retries_when_a_stream_is_stricter_than_it_declares():
    class LyingStream:
        encoding = "utf-8"

        def __init__(self) -> None:
            self.written: list[str] = []

        def write(self, text: str) -> None:
            if any(ord(character) > 127 for character in text):
                raise UnicodeEncodeError("ascii", text, 0, 1, "not ascii")
            self.written.append(text)

    stream = LyingStream()
    write_safe(stream, "\u9ad8\u538b")
    assert stream.written == ["\\u9ad8\\u538b"]


def test_write_line_appends_exactly_one_newline():
    stream = io.StringIO()
    write_line(stream, "row")
    write_line(stream)
    assert stream.getvalue() == "row\n\n"


# --- the real command line ---------------------------------------------------


def test_cli_list_exits_zero_with_strict_cp1252_stdout(tmp_path):
    game = _game_with_saves(tmp_path)
    environment = dict(os.environ, PYTHONIOENCODING="cp1252")
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "prepare_capture_dataset.py"),
         "--game-dir", str(game), "--list"],
        cwd=ROOT, capture_output=True, env=environment, timeout=180)
    assert completed.returncode == 0, completed.stderr.decode("cp1252", "replace")
    rendered = completed.stdout.decode("cp1252")
    assert len(rendered.splitlines()) == 5
    assert CJK_ESCAPED in rendered
    assert CJK_SAVE_ESCAPED in rendered
    assert "Caf\u00e9 World" in rendered
    # No lossy "?" substitution: either the character survived or it was escaped.
    assert "????" not in rendered
