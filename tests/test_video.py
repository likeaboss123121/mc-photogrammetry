import pytest

from mc_photogrammetry.input.video import VideoDecodeError, iter_video_frames


def test_video_sampling_validates_sample_rate():
    with pytest.raises(ValueError, match="sample_every"):
        list(iter_video_frames("capture.mp4", sample_every=0))


def test_video_decoder_reports_missing_optional_backend(tmp_path):
    source = tmp_path / "capture.mp4"
    source.write_bytes(b"not a video")

    try:
        frames = list(iter_video_frames(source))
    except VideoDecodeError as error:
        assert "could not open video" in str(error) or "requires opencv" in str(error)
    else:
        assert frames == []
