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
from .lattice import rectified_pixel_step
from .lattice_planes import (
	DistanceAlternative,
	PeriodicEnergyPhaseEvidence,
	PlaneReading,
	PlaneRectification,
	PlaneSignature,
	frame_plane_readings,
	frame_plane_readings_from_proposals,
	plane_signature,
	rectification_homography,
	warp_signature,
)
from .plane_graph import (
	ComponentSolution,
	PlaneMatch,
	axis_transform,
	cell_grid_homography,
	class_constrained_bestof,
	match_plane_graph,
	match_plane_pair,
	resolved_class,
	solve_component,
)
from .plane_axis_support import (
	AxisSupportComponent,
	C1AxisSupport,
	C1MatchIdentity,
	FrameAxisSupport,
	MappedCellShiftAxis,
	MatchSupportRefusal,
	NormalFaceConstraint,
	build_axis_support,
)
from .region_growth import (
	DEFAULT_REGION_GROWTH_CONFIG,
	RegionGrowthConfig,
	propose_image_planes_v2,
)

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
	"rectified_pixel_step",
	"DistanceAlternative",
	"PeriodicEnergyPhaseEvidence",
	"PlaneReading",
	"PlaneRectification",
	"PlaneSignature",
	"frame_plane_readings",
	"frame_plane_readings_from_proposals",
	"plane_signature",
	"rectification_homography",
	"warp_signature",
	"ComponentSolution",
	"PlaneMatch",
	"axis_transform",
	"cell_grid_homography",
	"AxisSupportComponent",
	"C1AxisSupport",
	"C1MatchIdentity",
	"FrameAxisSupport",
	"MappedCellShiftAxis",
	"MatchSupportRefusal",
	"NormalFaceConstraint",
	"build_axis_support",
	"class_constrained_bestof",
	"match_plane_graph",
	"match_plane_pair",
	"resolved_class",
	"solve_component",
	"DEFAULT_REGION_GROWTH_CONFIG",
	"RegionGrowthConfig",
	"propose_image_planes_v2",
]
