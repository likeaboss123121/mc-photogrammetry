"""Block-grid reconstruction primitives."""

from .grid import EvidenceGrid, FillPolicy, evidence_to_volume
from .facade import DEFAULT_PALETTE, PaletteColor, estimate_horizon, reconstruct_facade
from .camera import CameraModel, load_camera_manifest, project_points, unproject_pixel
from .multiview import (
	VisibleBlock,
	fuse_calibrated_views,
	reconstruct_visible_surfaces,
	visible_blocks_for_camera,
)
from .scene import SceneTransform, load_scene_transform
from .texture_calibration import AppearanceCalibration, calibrate_appearance
from .sfm import SfmSettings, reconstruct_from_calibrated_views
from .raycast import DEFAULT_FAMILY_PRIORS, reconstruct_scene_surfaces

__all__ = [
	"DEFAULT_PALETTE",
	"CameraModel",
	"load_camera_manifest",
	"EvidenceGrid",
	"FillPolicy",
	"PaletteColor",
	"evidence_to_volume",
	"estimate_horizon",
	"reconstruct_facade",
	"project_points",
	"unproject_pixel",
	"VisibleBlock",
	"fuse_calibrated_views",
	"reconstruct_visible_surfaces",
	"visible_blocks_for_camera",
	"SceneTransform",
	"load_scene_transform",
	"AppearanceCalibration",
	"calibrate_appearance",
	"SfmSettings",
	"reconstruct_from_calibrated_views",
	"reconstruct_scene_surfaces",
	"DEFAULT_FAMILY_PRIORS",
]
