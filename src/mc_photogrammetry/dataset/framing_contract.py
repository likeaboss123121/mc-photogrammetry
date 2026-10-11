"""Shared evaluator/planner projected-occupancy contract defaults."""

# Keep these frozen promotion floors in one lightweight module so planning can
# honor evaluator defaults without importing the artifact-reading QA module.
MIN_PROJECTED_OCCUPANCY_WIDTH = 0.30
MIN_PROJECTED_OCCUPANCY_HEIGHT = 0.30
MAX_PROJECTED_REGION_ENVELOPE = 0.85
MIN_CAMERA_CLEARANCE = 2.0
