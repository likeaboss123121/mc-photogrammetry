"""Focused tests for unattended-capture client option staging (DS-22).

The interactive capture client runs on a hidden Windows desktop, so preparation
must stage three safe client defaults into the staging game directory without
disturbing any other client option:

* ``pauseOnLostFocus:false`` so an unfocused window does not pause the world;
* ``soundCategory_master:0.0`` so the client starts muted;
* ``rawMouseInput:false`` so the client never grabs or steers the mouse.

Everything here is Minecraft-free and uses public helpers only.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from mc_photogrammetry.dataset import (
    CAPTURE_OPTIONS,
    OPTIONS_FILE_NAME,
    PAUSE_ON_LOST_FOCUS,
    RAW_MOUSE_INPUT,
    SOUND_CATEGORY_MASTER,
    DatasetSchemaError,
    options_path,
    render_options,
    stage_capture_options,
)

ROOT = Path(__file__).resolve().parents[1]

#: The three staged lines in the order the renderer appends absent keys.
STAGED_LINES = [
    "pauseOnLostFocus:false",
    "rawMouseInput:false",
    "soundCategory_master:0.0",
]
STAGED_NEWLINE = "\n".join(STAGED_LINES) + "\n"


def _game(tmp_path: Path) -> Path:
    game = tmp_path / "game"
    game.mkdir(parents=True, exist_ok=True)
    return game


def test_capture_option_set_is_exactly_the_focus_mute_and_mouse_guards():
    assert dict(CAPTURE_OPTIONS) == {
        PAUSE_ON_LOST_FOCUS: "false",
        SOUND_CATEGORY_MASTER: "0.0",
        RAW_MOUSE_INPUT: "false",
    }
    assert set(CAPTURE_OPTIONS) == {
        PAUSE_ON_LOST_FOCUS, SOUND_CATEGORY_MASTER, RAW_MOUSE_INPUT}
    assert options_path("g") == Path("g") / OPTIONS_FILE_NAME


def test_absent_options_file_is_created_with_only_the_capture_guards(tmp_path):
    game = _game(tmp_path)
    path = stage_capture_options(game)
    assert path == game / "options.txt"
    assert path.read_bytes() == STAGED_NEWLINE.encode("utf-8")


def test_existing_options_survive_and_every_guard_is_replaced_in_place(tmp_path):
    game = _game(tmp_path)
    existing = "\n".join([
        "version:4189",
        "fov:0.0",
        "soundCategory_master:1.0",
        "gamma:0.5",
        "guiScale:3",
        "pauseOnLostFocus:true",
        "rawMouseInput:true",
        "renderDistance:12",
        "lang:en_us",
    ]) + "\n"
    (game / "options.txt").write_bytes(existing.encode("utf-8"))

    stage_capture_options(game)

    lines = (game / "options.txt").read_text(encoding="utf-8").splitlines()
    assert lines == ["version:4189", "fov:0.0", "soundCategory_master:0.0",
                     "gamma:0.5", "guiScale:3", "pauseOnLostFocus:false",
                     "rawMouseInput:false", "renderDistance:12", "lang:en_us"]
    # Only the three guard values changed: same lines, same positions.
    assert (game / "options.txt").read_bytes() == (
        existing
        .replace("soundCategory_master:1.0", "soundCategory_master:0.0")
        .replace("pauseOnLostFocus:true", "pauseOnLostFocus:false")
        .replace("rawMouseInput:true", "rawMouseInput:false")
    ).encode("utf-8")


def test_existing_options_without_guards_gain_them_appended_in_sorted_order(tmp_path):
    game = _game(tmp_path)
    existing = "\n".join([
        "version:4189",
        "fov:0.0",
        "gamma:0.5",
        "pauseOnLostFocus:true",
        "renderDistance:12",
    ]) + "\n"
    (game / "options.txt").write_bytes(existing.encode("utf-8"))

    stage_capture_options(game)

    lines = (game / "options.txt").read_text(encoding="utf-8").splitlines()
    assert lines == ["version:4189", "fov:0.0", "gamma:0.5",
                     "pauseOnLostFocus:false", "renderDistance:12",
                     "rawMouseInput:false", "soundCategory_master:0.0"]


def test_staging_twice_is_byte_identical_and_never_keeps_unsafe_values(tmp_path):
    game = _game(tmp_path)
    (game / "options.txt").write_bytes(
        b"fov:0.0\npauseOnLostFocus:true\nsoundCategory_master:0.7\n"
        b"rawMouseInput:true\ngamma:0.5\n")

    first = stage_capture_options(game).read_bytes()
    assert first == (
        b"fov:0.0\npauseOnLostFocus:false\nsoundCategory_master:0.0\n"
        b"rawMouseInput:false\ngamma:0.5\n")
    for unsafe in (b"pauseOnLostFocus:true", b"rawMouseInput:true",
                   b"soundCategory_master:0.7"):
        assert unsafe not in first

    second = stage_capture_options(game).read_bytes()
    assert second == first


def test_duplicate_keys_collapse_to_the_first_position(tmp_path):
    game = _game(tmp_path)
    (game / "options.txt").write_bytes(
        b"pauseOnLostFocus:true\nsoundCategory_master:0.5\nfov:0.0\n"
        b"pauseOnLostFocus:true\nsoundCategory_master:1.0\nrawMouseInput:true\n")

    stage_capture_options(game)

    lines = (game / "options.txt").read_text(encoding="utf-8").splitlines()
    assert lines == ["pauseOnLostFocus:false", "soundCategory_master:0.0",
                     "fov:0.0", "rawMouseInput:false"]
    for key in (PAUSE_ON_LOST_FOCUS, SOUND_CATEGORY_MASTER, RAW_MOUSE_INPUT):
        matches = [line for line in lines if line.partition(":")[0] == key]
        assert matches == [f"{key}:{dict(CAPTURE_OPTIONS)[key]}"]


def test_missing_keys_are_appended_once_and_stay_single(tmp_path):
    game = _game(tmp_path)
    (game / "options.txt").write_bytes(b"fov:0.0\n")
    path = stage_capture_options(game)
    assert path.read_bytes() == f"fov:0.0\n{STAGED_NEWLINE}".encode("utf-8")

    assert stage_capture_options(game).read_bytes() == (
        f"fov:0.0\n{STAGED_NEWLINE}".encode("utf-8"))


def test_line_endings_and_trailing_newline_are_preserved(tmp_path):
    crlf = _game(tmp_path / "crlf")
    (crlf / "options.txt").write_bytes(
        b"fov:0.0\r\npauseOnLostFocus:true\r\nsoundCategory_master:0.4\r\n")
    stage_capture_options(crlf)
    assert (crlf / "options.txt").read_bytes() == (
        b"fov:0.0\r\npauseOnLostFocus:false\r\nsoundCategory_master:0.0\r\n"
        b"rawMouseInput:false\r\n")

    bare = _game(tmp_path / "no-trailing-newline")
    (bare / "options.txt").write_bytes(b"fov:0.0")
    stage_capture_options(bare)
    assert (bare / "options.txt").read_bytes() == (
        b"fov:0.0\npauseOnLostFocus:false\nrawMouseInput:false\n"
        b"soundCategory_master:0.0")


def test_render_options_is_a_pure_function_of_its_input():
    assert render_options(None) == STAGED_NEWLINE
    assert render_options("") == STAGED_NEWLINE
    assert render_options(None) == render_options(render_options(None))
    assert render_options("fov:0.0") == "fov:0.0\n" + STAGED_NEWLINE.rstrip("\n")
    assert render_options(render_options("fov:0.0")) == render_options("fov:0.0")


def test_staging_refuses_a_directory_at_the_options_path(tmp_path):
    game = _game(tmp_path)
    (game / "options.txt").mkdir()
    with pytest.raises(DatasetSchemaError, match="directory"):
        stage_capture_options(game)


def _legacy_level_dat(path: Path, *, level_name: str, spawn_x: int = 3, spawn_z: int = -4,
                      spawn_y: int = 71) -> None:
    import nbtlib
    from nbtlib import tag

    data = {
        "LevelName": tag.String(level_name),
        "LastPlayed": tag.Long(1_700_000_000_000),
        "SpawnX": tag.Int(spawn_x),
        "SpawnY": tag.Int(spawn_y),
        "SpawnZ": tag.Int(spawn_z),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    nbtlib.File(tag.Compound({"Data": tag.Compound(data)}), gzipped=True).save(path)


def _prepare_module():
    spec = importlib.util.spec_from_file_location(
        "ds22_prepare_capture_dataset", ROOT / "scripts" / "prepare_capture_dataset.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_capture_dataset_stages_every_guard_in_the_staging_game(tmp_path, capsys):
    game = tmp_path / "game"
    _legacy_level_dat(game / "saves" / "Alpha" / "level.dat", level_name="Alpha World")
    region = game / "saves" / "Alpha" / "region"
    region.mkdir(parents=True, exist_ok=True)
    (region / "r.0.0.mca").write_bytes(b"region")

    dataset = tmp_path / "dataset"
    # An earlier run's staging game directory already holds unrelated options and
    # unsafe guard values; preparation keeps the unrelated options, replaces the
    # unsafe values in place, and appends the missing guard.
    staging = dataset / "staging-game"
    staging.mkdir(parents=True)
    (staging / "options.txt").write_bytes(
        b"fov:0.0\r\npauseOnLostFocus:true\r\nsoundCategory_master:0.8\r\ngamma:0.5\r\n")

    prepare = _prepare_module()
    exit_code = prepare.main([
        "--game-dir", str(game), "--save", "Alpha", "--dataset-id", "ds-test",
        "--seed", "7", "--landscape", "--radius", "4", "--views", "4",
        "--dataset", str(dataset),
    ])
    assert exit_code == 0
    out = capsys.readouterr().out
    for guard in STAGED_LINES:
        assert guard in out
    assert (staging / "options.txt").read_bytes() == (
        b"fov:0.0\r\npauseOnLostFocus:false\r\nsoundCategory_master:0.0\r\n"
        b"gamma:0.5\r\nrawMouseInput:false\r\n")
