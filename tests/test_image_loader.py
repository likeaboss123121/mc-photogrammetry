from pathlib import Path

import pytest

from mc_photogrammetry.input.image import ImageDecodeError, load_image


def test_image_loader_rejects_non_image_extensions(tmp_path):
    source = tmp_path / "capture.txt"
    source.write_text("not an image", encoding="utf-8")

    with pytest.raises(ImageDecodeError, match="unsupported image format"):
        load_image(source)


def test_image_loader_requires_an_existing_file(tmp_path):
    with pytest.raises(ImageDecodeError):
        load_image(Path(tmp_path) / "missing.png")
