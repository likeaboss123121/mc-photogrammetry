from pathlib import Path

from mc_photogrammetry.input.media import MediaKind, discover_media


def test_discover_media_is_sorted_and_deduplicates_images(tmp_path):
    image_a = tmp_path / "b.png"
    image_b = tmp_path / "a.jpg"
    duplicate = tmp_path / "copy.webp"
    video = tmp_path / "orbit.mp4"
    image_a.write_bytes(b"same image")
    image_b.write_bytes(b"other image")
    duplicate.write_bytes(b"same image")
    video.write_bytes(b"video placeholder")

    manifest = discover_media([tmp_path])

    assert [frame.kind for frame in manifest.frames] == [MediaKind.IMAGE, MediaKind.IMAGE, MediaKind.VIDEO]
    assert [Path(frame.source_path).name for frame in manifest.frames] == ["a.jpg", "b.png", "orbit.mp4"]
    assert len(manifest.image_frames) == 2
    assert len(manifest.video_sources) == 1


def test_discover_media_rejects_missing_paths(tmp_path):
    missing = tmp_path / "missing.png"

    try:
        discover_media([missing])
    except FileNotFoundError as error:
        assert error.args == (missing,)
    else:
        raise AssertionError("missing media should be rejected")
