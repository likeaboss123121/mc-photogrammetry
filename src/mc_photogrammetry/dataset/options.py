"""Stage the client ``options.txt`` that a real-Minecraft capture needs.

The capture client runs on a hidden, unfocused Windows desktop, where three
vanilla client defaults are hostile to an unattended capture:

* ``pauseOnLostFocus:true`` pauses an integrated singleplayer world as soon as
  the window loses focus, so the pause menu would sit over the build and every
  later frame would record UI instead of geometry;
* a live master sound category can play game audio through the capture machine;
* ``rawMouseInput:true`` makes the client capture/steer the mouse, which steals
  the operator's pointer (and any input) on the shared desktop.

Preparation therefore stages ``pauseOnLostFocus:false``,
``soundCategory_master:0.0``, and ``rawMouseInput:false`` into the staging game
directory before the mod is armed. The client must start silent and must not
grab the mouse.

The staging is deliberately minimal and lossless:

* only the named capture keys are rewritten; every other line of an existing
  ``options.txt`` is preserved in its original order, so the deterministic
  capture settings the plan relies on (FOV, gamma, HUD, packs, render distance,
  ...) are never silently changed by preparation;
* a missing file becomes a minimal file that carries only the staged keys, so
  the client's own defaults still apply to everything else;
* running twice is idempotent and can never grow a duplicate key: the first
  occurrence keeps its position and receives the staged value, later occurrences
  are dropped, and absent keys are appended in sorted order.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from mc_photogrammetry.dataset.schemas import DatasetSchemaError

#: The vanilla client option that pauses an integrated server when the window
#: loses focus.
PAUSE_ON_LOST_FOCUS = "pauseOnLostFocus"

#: The vanilla client master sound category; capture clients must start muted.
SOUND_CATEGORY_MASTER = "soundCategory_master"

#: The vanilla client option that lets Minecraft grab/steer the physical mouse;
#: a capture client must never steal the operator's input.
RAW_MOUSE_INPUT = "rawMouseInput"

#: The capture-time option set: an unfocused client keeps running, starts muted,
#: and does not take the mouse. Values are strings because ``options.txt`` is
#: line-oriented text.
CAPTURE_OPTIONS: Mapping[str, str] = {
    PAUSE_ON_LOST_FOCUS: "false",
    SOUND_CATEGORY_MASTER: "0.0",
    RAW_MOUSE_INPUT: "false",
}

OPTIONS_FILE_NAME = "options.txt"


def options_path(game_dir: Path | str) -> Path:
    """Where the client reads and writes its options inside a game directory."""
    return Path(game_dir) / OPTIONS_FILE_NAME


def _newline_for(text: str) -> str:
    """The line ending an existing file already uses, defaulting to ``\\n``."""
    return "\r\n" if "\r\n" in text else "\n"


def _option_key(line: str) -> str:
    """The key half of one ``key:value`` options line."""
    return line.partition(":")[0].strip()


def _validate_override(key: str, value: str) -> None:
    if not key or any(character in key for character in (":", "\r", "\n")):
        raise DatasetSchemaError(f"unsafe option key: {key!r}")
    if any(character in value for character in ("\r", "\n")):
        raise DatasetSchemaError(f"option {key!r} value must be a single line")


def render_options(existing: str | None, *,
                   overrides: Mapping[str, str] | None = None) -> str:
    """Return ``options.txt`` content with ``overrides`` applied, everything else kept.

    The first occurrence of every overridden key keeps its position and receives
    the staged value; later occurrences are dropped, so the result can never carry
    a duplicate key. Keys that are absent are appended in sorted order, which
    makes the output a pure function of the input. An absent or empty input gains
    the usual trailing newline; an existing file keeps whatever its last line did.
    """
    raw_settings = CAPTURE_OPTIONS if overrides is None else overrides
    settings: dict[str, str] = {}
    for raw_key, raw_value in raw_settings.items():
        key, value = str(raw_key), str(raw_value)
        _validate_override(key, value)
        settings[key] = value

    text = "" if existing is None else existing
    newline = _newline_for(text)
    trailing = existing is None or text == "" or text.endswith(("\n", "\r"))
    lines: list[str] = []
    applied: set[str] = set()
    for line in text.splitlines():
        key = _option_key(line)
        if key not in settings:
            lines.append(line)
            continue
        if key in applied:
            continue
        lines.append(f"{key}:{settings[key]}")
        applied.add(key)
    for key in sorted(settings):
        if key not in applied:
            lines.append(f"{key}:{settings[key]}")
    rendered = newline.join(lines)
    if trailing:
        rendered += newline
    return rendered


def stage_capture_options(game_dir: Path | str, *,
                          overrides: Mapping[str, str] | None = None) -> Path:
    """Write the capture option set into ``<game_dir>/options.txt`` and return it.

    Existing unrelated options are preserved and the file is rewritten in place,
    so re-running preparation against an already-staged directory is idempotent.
    The bytes are read and written without newline translation, so a client file
    using CRLF stays CRLF.
    """
    path = options_path(game_dir)
    if path.exists() and path.is_dir():
        raise DatasetSchemaError(f"options path is a directory, not a file: {path}")
    existing = path.read_bytes().decode("utf-8", "surrogateescape") if path.exists() else None
    rendered = render_options(existing, overrides=overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(rendered.encode("utf-8", "surrogateescape"))
    return path
