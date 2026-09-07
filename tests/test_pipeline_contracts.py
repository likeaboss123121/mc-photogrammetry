from pathlib import Path

import numpy as np

from mc_photogrammetry.domain.models import BlockVolume
from mc_photogrammetry.pipeline.contracts import (
    CameraEstimate,
    CameraIntrinsics,
    CameraPose,
    CameraSolution,
    FrameObservation,
    FrameTransform,
    ReconstructionResult,
    RenderResult,
)


def test_frame_transform_defaults_to_the_original_image_shape():
    transform = FrameTransform((1920, 1080))
    frame = FrameObservation("frame-1", Path("capture.png"), transform)

    assert frame.transform.resolved_crop_size == (1920, 1080)
    assert frame.transform.resolved_output_size == (1920, 1080)


def test_camera_and_render_contracts_preserve_explicit_geometry_metadata():
    camera = CameraEstimate(
        CameraIntrinsics((1280, 720), (600.0, 600.0), (640.0, 360.0)),
        CameraPose((1.0, 2.0, 3.0), 15.0, -5.0),
        residual_pixels=0.8,
        confidence=0.9,
    )
    solution = CameraSolution({"frame-1": camera}, (("frame-1",),), scale_known=False)
    render = RenderResult(np.zeros((2, 2, 3)), np.ones((2, 2)), np.zeros((2, 2), dtype=int))
    result = ReconstructionResult(BlockVolume(), complete=False)

    assert solution.cameras["frame-1"].intrinsics.principal_point_pixels == (640.0, 360.0)
    assert render.depth.shape == render.ownership.shape
    assert not result.complete
