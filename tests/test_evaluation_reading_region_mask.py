from dataclasses import replace
import numpy as np
import pytest

from mc_photogrammetry import evaluation_reading_region_mask as subject
from mc_photogrammetry.reconstruction import lattice_planes as lp


def _reading():
    mask = np.zeros((12, 16), dtype=bool)
    mask[2:9, 3:12] = True
    mask[4:6, 6:9] = False
    polygon = ((3.0, 2.0), (11.0, 2.0), (11.0, 8.0), (3.0, 8.0))
    evidence = lp._make_region_evidence(mask, 0, polygon, ("proposal-a",))
    reading = lp.PlaneReading(
        "frame-a", 0, 1.0, (lp.DistanceAlternative(1.0, 2.0, 1.0),),
        4.0, (0.0, 0.0), 3.0, 16, polygon, 20, .4, .8,
        lp.PlaneRectification(((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)),
                              100., (8., 6.), (8., 6.)), evidence)
    return reading, mask


def test_exact_decode_equivalence_crop_holes_and_readonly():
    reading, expected = _reading()
    result = subject.reading_region_mask(reading, canonical_reading_index=3)
    assert np.array_equal(result.mask, expected)
    assert not result.mask.flags.writeable
    assert result.height == 12 and result.width == 16
    assert result.source_region_id == reading.region_evidence.source_region_id


def test_optional_gui_usable_mask_is_exact_and_resize_refused():
    reading, expected = _reading()
    usable = np.ones(expected.shape, dtype=bool)
    usable[:4] = False
    result = subject.reading_region_mask(
        reading, canonical_reading_index=0, usable_mask=usable)
    assert np.array_equal(result.mask, expected & usable)
    with pytest.raises(ValueError, match="dimensions"):
        subject.reading_region_mask(
            reading, canonical_reading_index=0,
            usable_mask=np.ones((6, 8), dtype=bool))


def test_digest_replay_and_tampering():
    result = subject.reading_region_mask(_reading()[0], canonical_reading_index=1)
    replay = subject.ReadingRegionMask(**{
        name: getattr(result, name) for name in result.__dataclass_fields__})
    assert replay == result
    with pytest.raises(ValueError, match="digest"):
        replace(result, frame_id="other")
    with pytest.raises(ValueError, match="malformed"):
        replace(result, mask_bool_bytes=result.mask_bool_bytes[:-1], mask_sha256="")


def test_provenance_mismatch_missing_evidence_and_caps():
    reading, _ = _reading()
    with pytest.raises(ValueError, match="frame provenance"):
        subject.reading_region_mask(
            reading, canonical_reading_index=0, expected_frame_id="foreign")
    with pytest.raises(ValueError, match="source-region provenance"):
        subject.reading_region_mask(
            reading, canonical_reading_index=0,
            expected_source_region_id="0" * 64)
    with pytest.raises(ValueError, match="pixel cap"):
        subject.reading_region_mask(reading, canonical_reading_index=0, max_pixels=100)
    with pytest.raises(ValueError, match="no region evidence"):
        subject.reading_region_mask(
            replace(reading, region_evidence=None), canonical_reading_index=0)


def test_helper_has_no_production_pipeline_or_io_coupling():
    text = open(subject.__file__, encoding="utf-8").read()
    assert "mc_photogrammetry.pipeline" not in text
    assert "pathlib" not in text and "subprocess" not in text
