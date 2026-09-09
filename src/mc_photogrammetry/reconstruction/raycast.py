from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from threading import Event

import numpy as np

from mc_photogrammetry.domain.models import BlockCoordinate, BlockObservation, BlockState, BlockVolume
from mc_photogrammetry.blocks.model_atlas import FACE_SHADE, FACE_SLOTS
from mc_photogrammetry.vision.gui_mask import create_gui_mask
from .camera import CameraModel, project_points
from .face_sampling import (
    DEPTH_SCALE,
    MINIMUM_FACE_AREA,
    PATCH_SIZE,
    rectify_face,
    sample_visible_faces,
)
from .facade import DEFAULT_PALETTE, PaletteColor
from .lighting import (
    emission_of,
    estimate_light_levels,
    locate_sources,
    LightField, LightResponse, ambient_occlusion, fit_response, light_field,
)
from .scene import SceneTransform
from .texture_calibration import AppearanceCalibration, calibrate_appearance_from_palette

OUTLIER_CEILING = 24.0
"""Cap on how much worse than its own best match a candidate may score in one view."""

CHROMA_WEIGHT = 0.75
"""Share of the appearance distance carried by hue rather than brightness.

Block lighting scales a face's brightness far more than it moves its hue, and
the texture term already carries the pattern, so hue is what colour is left to
contribute. Anything from 0.8 to 1.0 measures within a point of the best on
Template 2; the value is kept below 1.0 deliberately, because a scene built from
greys - stone, concrete, wool - has no hue to separate and needs the brightness
term that a weight of 1.0 throws away.
"""

MINIMUM_LIGHT = 1.0
"""Darkest a face may be assumed lit, as a fraction of its full-bright colour."""

OCCLUSION_STRENGTH = 1.0
"""How much of vanilla's ambient occlusion to undo before matching.

One scales the modelled shading straight off the observation; zero ignores
it. Corners and ceilings are darker than open wall of the same material,
and a pale block in a corner otherwise reads as a darker one.
"""

SIMPLICITY_WEIGHT = 3.0
SHAPE_WEIGHT = 6.0
"""Cost for a candidate whose shape does not fill the face being looked at.

Admitting partial blocks to the palette without this put a stair, slab or carpet
on 760 of Template 2's 814 cells: they are cut from the same texture as the full
block, so they match a face exactly as well, and the palette holds twenty-four
stair orientations against one entry for planks.
"""
"""Cost per distinct face a candidate wears beyond the second.

Settles ties in favour of plain building material over fixtures that
borrow the same wood or stone on their sides.
"""

DIM_SCENE_THRESHOLD = 0.85
ADJACENCY_STRENGTH = 0.55
"""How strongly a block is pulled toward what its neighbours decided.

The library defaulted to 0.0 - adjacency off - while `benchmark_templates.py`
passed 0.55 and every measurement in the handoff used 1.0, so nobody calling the
library or the CLI without the flag was getting a configuration anyone had
measured. Off is the worst of the three on both metrics: Template 2 scores 73.72%
exact state against 85.64% at 0.55, and 17% more visible material error.

The landscape is sharp and not smooth - 0.55 gives 85.64% and 0.60 gives 78.85% -
so treat this as a measured operating point rather than a robust optimum, and
re-sweep it after any change to the palette or the classifier.
"""
PROPOSAL_LIMIT = 40
"""Cells the residual pass may propose.

Twenty, while the fit named the right block only a quarter of the time: past that
each further proposal was a block placed where nothing needed one, and geometry F1
fell from 94.14% through 93.62% to 92.38% for no gain in exact state.

Giving the fit texture changed the trade, because a proposal is now worth roughly
twice what it was. Re-measured on Template 2, against 668 correct blocks and
93.64% F1 with the chain off: twenty gives 672 and 93.59%, forty gives 672 and
**93.67%**, eighty gives 675 and 92.90%. Forty is the only setting that beats the
baseline on both counts at once, so it is no longer a trade at all - the extra
twenty proposals correct as much geometry as they cost. Eighty buys three more
blocks for three quarters of a point of F1 and nearly twice the runtime.
"""
FIT_PARTIAL_SHAPES = False
"""Re-decide proposed cells by fitting a partial shape against the photographs.

The classifier compares a rectified face to a texture, which cannot tell a stair
from the planks it is cut from. `shape_fit` renders each candidate into the scene
instead. Off by default until the proposal source that feeds it is good enough to
be worth the runtime.
"""
STEREO_INTERIOR = False
"""Recover interior cells by stereo along the ray.

Off by default. The cells it finds are mostly right - 75-98% on Templates 4 and 6,
against 8-23% for every other method tried - and it improves geometry (Template 4
+1.51pp, Template 6 +0.19pp). But adding them collapses exact-state accuracy on
Template 6 from 30.37% to 3.40%, for reasons not yet understood: thirty-one extra
cells cannot do that directly, so something downstream - palette pruning, the
support prior or adjacency - is being destabilised by them. Not shippable until
that is found.
"""
STEREO_MARGIN = 0.25
STEREO_SCALE = 4
CORRECT_LIGHT_CAST = True
"""Undo the image-derived light cast before classifying dim captures.

The previous self-labelled brightness gate classified Template 6's uniformly dim
walls as dark materials and reported a scene brightness of 0.983. Its palette
comparison is circular: a broad palette always contains a dark look-alike. The
chroma-spread gate is independent of candidate labels. Under identical current
geometry it restores Template 6 from 0 to 613 exact states while leaving the
fullbright Template 2 run unchanged (655 exact states).
"""
DEFAULT_PARTIAL_FIT_MAX_RENDERS = 256
"""Finite default candidate-render budget for developer partial fitting."""
DIM_SPREAD_THRESHOLD = 1.9
"""Brightness spread above which a scene is treated as lit rather than fullbright."""
LIGHT_CAST = (1.234, 1.002, 0.763)
"""Normalised per-channel gain of Minecraft block light, measured on three fixtures."""
LIGHT_CAST_DAMPING = 1.0
"""How much of the recovered dimming to undo; below one deliberately under-corrects."""
CHROMA_BINS = 12
CAST_REGION = 0
"""Edge, in blocks, of the box a colour's unlit brightness is estimated over.

Zero means one estimate for the whole scene. 54 measured region 16 and 8 as worse
*on a twenty-block crop*, which was the wrong place to test it: a crop is already
local, so binning inside it only starves each group. The full fixture is the case
that needs it, and 57 is where that is measured.
"""
CAST_PERCENTILE = 99.0
"""Percentile of a colour's face brightness taken as its unlit appearance.

The maximum is the obvious choice and it is not scale-invariant: over a twenty-cube
crop it is taken across four thousand faces and over the whole of Template 5 across
ninety thousand, so a larger scene's is systematically higher and over-corrects.

Measured on Template 5's diamond crop, which reads 0.00% without any correction:
100 gives 24.33%, 99.5 gives 28.39%, **99.0 gives 48.44%**, 98 gives 46.93%, and 97
collapses back to 0.00%. The window is narrow and the optimum is sharp. Template 6
prefers it slightly (0.6650 against 0.6768 relative error) and Template 2 is
untouched, its gate correctly declining.
"""
CAST_GROUP_MINIMUM = 12
CAST_FLOOR = 0.25
"""Below this brightness a scene is treated as lit and worth un-lighting.

Measured on the faces the classifier chose for itself, where the leftover
brightness is honest. Zero disables the test.
"""

AMBIENT_LEVELS: tuple[int, ...] = ()
"""Floors to try under the simulated light field, chosen by best fit.

Empty disables the search. A floor makes the field explain the faces far
better - Template 6's cost falls from 120 to 37 - and yet classifies worse,
30.51% against 20.32%, because it undoes the very un-lighting that fixes a
dim scene. Best-match cost is not a proxy for being right: a darker block
always exists, so the cost always prefers leaving an observation alone.
"""

GLOBAL_EXPOSURE = True
"""Correct a dim scene by how far under the jar it measures overall.

A light field carries this only where it has located sources. A room lit
from outside the crop, or by more sources than were found, is uniformly
dim and wants a uniform correction.
"""

SLOT_RESPONSE = True
"""Correct for light arriving more from one direction than another.

A light field gives a cell one level and every face of it shares that level,
so a directional bias beyond vanilla shading survives it.
"""

SOURCE_CANDIDATES = 6
"""How many light sources to locate before rebuilding the field."""

TRILATERATE_SOURCES = True
"""Locate light sources from the brightness they produce.

Minecraft light falls exactly one level per block, so an observed
brightness is a distance measurement and enough of them intersect at a
source's position - including sources no camera can see.
"""

LIGHT_MUST_HELP = False
"""Keep a fitted light field only when it explains the faces better.

Guards a fullbright scene against being dimmed by light sources detected in
error, without needing to be told which kind of scene it is.
"""

VANILLA_LIGHT_RESPONSE = True
"""Use Minecraft's own light curve instead of fitting one from the scene.

Correct whenever the client is at default gamma with no shaders, and the
only option that works in a dim scene, where a fitted response has no
correctly classified blocks to fit from.
"""

DETECT_EMITTERS = False
"""Whether to find light sources by their self-lit appearance.

A block that makes its own light renders at its jar colour whatever the
room's exposure is, so it can be recognised when nothing else can, and it
tells the lighting simulation where the sources are.
"""

EMITTER_BRIGHTNESS = 120.0
"""How bright an emitting candidate must be to be worth looking for."""

EMITTER_COST = 5.0
"""How closely a face must match a self-lit block to be called one."""

REQUIRE_RESOLVED_FACE = False
"""Whether a block is placed only where a face was genuinely resolved.

The centre-patch fallback reaches almost every cell, so leaving this off
fills the volume solid and leaves the enclosed-block pruning to carve it
back to the region boundary.
"""

PROPOSE_COLOUR_GAP = 6.0
"""How closely two views must agree in colour to propose an unseen cell.

Zero keeps the shipped behaviour. Positive values let the cameras propose
cells the occupancy pass never considers, which is the only way anything
standing inside a room - a pillar, a portal, a light source - is placed.
"""

PROPOSE_AGREEMENT = 0.5
"""Pattern agreement also required of a proposed cell."""

FIT_EXPOSURE = False
"""Whether a frame's calibration may carry overall brightness, not just cast.

Needed for scenes captured below full brightness, where no emitter inside
the crop can tell the lighting simulation how dark the room really was.
"""

EXPOSURE_FLOOR = 0.2
"""Darkest the palette may be scaled to meet an underexposed frame."""

EXPOSURE_CEILING = 3.0
"""Brightest the palette may be scaled to meet an overexposed frame."""

RECONCILE_EXPOSURE = True
"""Whether to put every frame on a common brightness scale before scoring.

Only the differences between frames are corrected; the scene's overall
brightness is left alone, because nothing in the images can determine it.
"""

OBLIQUITY_POWER = 0.0
"""How sharply an observation is discounted for being seen edge-on.

Zero trusts a grazing view as much as a face-on one. One weights it by the
cosine of the angle between the face normal and the camera.
"""

DISAGREEMENT_RISK = 0.0
"""Extra pull toward neighbours for a block its cameras disagree about.

Blocks the views agree on are measurably far more often right than blocks
they split over, and averaging a split leaves a muddle that is worth less
than what the surrounding blocks say.
"""

VISIBILITY_POWER = 0.0
"""How sharply an observation is discounted for being partly hidden.

Zero trusts a partly occluded face as much as a clear one. One weights it
by the share of it the camera can actually see.
"""

FALLBACK_RISK = 3.0
ADOPT_BOOST = 0.0
"""Coupling boost above which a cell may take its neighbours' label outright.

Adjacency can only ever re-rank a cell's *own* candidates, so a cell whose evidence
never proposed the right answer cannot be rescued by any amount of coupling. That
is exactly the centre-patch fallback cells - 6151 of Template 6's 7725 - and why
`fallback_risk` measured completely inert from 0 to 40. Above this boost the
neighbours' majority is admitted as a candidate in its own right.
"""
"""Extra pull toward neighbours for a block read from a centre patch.

Those blocks were never resolved as a face, so the pixels standing in for
them may belong to whatever is in front of them.
"""

MINIMUM_VISIBLE_FRACTION = 0.6
"""How much of a face must be unoccluded before it is worth rectifying.

A face below this is dropped and its block falls back to a centre patch,
which is a much poorer read, so the threshold trades one kind of
contamination against a coarser measurement.
"""

UPWARD_RISK = 11.0
"""Extra pull toward neighbours for a block seen only from above.

Upward faces are the ones carpets, slabs and stairs cover, so evidence
gathered from them is the most likely to describe a different block.
"""

EVIDENCE_POWER = 0.0
"""How strongly a block resists its neighbours per frame that saw it.

Zero treats every block's appearance evidence as equally trustworthy.
Higher values let thinly observed blocks follow their surroundings.
"""

AREA_POWER = 0.5
"""Exponent on face area when weighting an observation.

One half weights a face by its side length. Higher values concentrate a
block's decision in the frames that saw it largest.
"""

FIT_TOLERANCE = 0.0
"""Scale over which a face's own best match decides how much it is trusted.

Zero trusts every face equally. Smaller values discount faces that nothing
in the palette explains well, which is what a contaminated patch looks like.
"""

SUPPORT_WEIGHT = 0.0
"""Strength of the prior that a scene reuses the materials it already shows."""

FALLBACK_WEIGHT = 1.0
"""Weight given to a centre-patch reading of a block no face resolved.

These blocks are the ones nothing can see, so the pixel standing in for one
may belong to whatever is in front of it. Lowering this leaves the block
placed but lets its neighbours decide what it is.
"""

MARGIN_SUPPORT = False
"""Whether palette support counts how decisively a candidate won each face.

Counting bare wins favours a candidate that resembles everything slightly
over one that matches a few faces exactly.
"""

LOCAL_SUPPORT_REGION = 6
"""Side of the cube support is also counted within, in blocks."""

LOCAL_SUPPORT_KEEP = 0
"""Candidates kept per region on top of the global survivors.

Zero keeps the global-only behaviour. A material used for one corner of a
build cannot survive a ranking over the whole of it.
"""

LOCAL_SUPPORT_WINS = 2.0
"""Weight a candidate must win inside a region to be kept for it."""

PALETTE_MINIMUM = 6
"""Fewest candidates the support pruning may leave standing.

A floor stops a scene collapsing onto its single most common material, but
it also overrides the support threshold entirely once that threshold bites,
so it is the setting that really decides how many candidates survive.
"""

PALETTE_SUPPORT = 0.02
"""Share of observed faces a candidate must win to stay in the palette.

Measured after a full pass rather than assumed, so nothing decides in
advance what a scene may contain. Zero keeps every candidate.
"""

MINIMUM_LIGHT_FACTOR = 0.15
"""Darkest a face may be assumed lit when undoing lighting.

Dividing by anything smaller turns sensor noise in a nearly black patch
into a confident colour.
"""

MINIMUM_COLOUR_CAST = 0.04
"""How far a frame's white balance must sit from neutral before it is applied."""

MINIMUM_LIGHT_EFFECT = 0.3
"""How well simulated light must predict observed brightness to be applied.

Correlation between the two across the sampled faces. Below this the light
model is not explaining the picture and is left out.
"""

TEXTURE_WEIGHT = 40.0
"""Weight on matching the face's full texture pattern, not just its colour.

Correlation of the rectified patch against the candidate's own face texture,
after removing brightness. It separates blocks that share a colour but not a
grain, and it is what block lighting cannot disturb.
"""

UNIFORMITY_SCALE = 1e9
"""Quadrant disagreement at which a face sample is trusted half as much."""

MAX_FACE_VARIATION = 12.0
"""How far apart a face patch's quadrants may look before the sample is dropped.

Chosen where the retained samples' median colour error bottoms out (8.0 ungated,
6.4 here). Tightening it further keeps only flat patches, and a patch that landed
wholly inside the wrong block is perfectly flat, so quality falls away again.
"""

FACE_SHRINK = 1.0
"""Fraction of each face to sample, measured from its centre.

The whole face is worth taking while the projection is accurate. Shrink it only
when the poses are known to be loose, since a face that lands slightly off then
drags the neighbouring block into the sample.
"""

VIEW_TOLERANCE = 3.0
"""How sharply a frame's disagreement with the others reduces its weight."""

DEFAULT_FAMILY_PRIORS = {
    "cube": 1.0,
    "log": 1.0,
    "slab": 1.0,
    "stairs": 1.0,
    "carpet": 1.0,
    "fence": 1.0,
    "plant": 1.0,
    "utility": 1.0,
    "other": 1.0,
}


def reconstruct_scene_surfaces(
    frames: dict[str, np.ndarray],
    cameras: dict[str, CameraModel],
    scene: SceneTransform,
    *,
    palette: tuple[PaletteColor, ...] = DEFAULT_PALETTE,
    sample_radius: int = 2,
    min_views: int = 1,
    columns: int | None = None,
    rows: int | None = None,
    depth_layers: int | None = None,
    max_color_distance: float | None = None,
    min_palette_margin: float = 0.0,
    min_state_agreement: float = 0.0,
    min_surface_contrast: float = 0.0,
    appearance_calibration: AppearanceCalibration | None = None,
    appearance_calibrations: dict[str, AppearanceCalibration] | None = None,
    surface_only: bool = True,
    adjacency_strength: float = ADJACENCY_STRENGTH,
    adjacency_min_neighbors: int = 3,
    bounded_adjacency: bool = False,
    minimum_face_area: float = MINIMUM_FACE_AREA,
    area_power: float = AREA_POWER,
    evidence_power: float = EVIDENCE_POWER,
    upward_risk: float = UPWARD_RISK,
    minimum_visible_fraction: float = MINIMUM_VISIBLE_FRACTION,
    fallback_risk: float = FALLBACK_RISK,
    adopt_boost: float = ADOPT_BOOST,
    fit_exposure: bool = FIT_EXPOSURE,
    require_resolved_face: bool = REQUIRE_RESOLVED_FACE,
    detect_emitters: bool = DETECT_EMITTERS,
    vanilla_light_response: bool = VANILLA_LIGHT_RESPONSE,
    light_must_help: bool = LIGHT_MUST_HELP,
    trilaterate_sources: bool = TRILATERATE_SOURCES,
    source_candidates: int = SOURCE_CANDIDATES,
    slot_response: bool = SLOT_RESPONSE,
    global_exposure: bool = GLOBAL_EXPOSURE,
    ambient_levels: tuple[int, ...] = AMBIENT_LEVELS,
    dim_scene_threshold: float = DIM_SCENE_THRESHOLD,
    correct_light_cast: bool = CORRECT_LIGHT_CAST,
    stereo_interior: bool = STEREO_INTERIOR,
    stereo_margin: float = STEREO_MARGIN,
    stereo_scale: int = STEREO_SCALE,
    extra_proposals: set[BlockCoordinate] | None = None,
    fit_partial_shapes: bool = FIT_PARTIAL_SHAPES,
    shape_palette: tuple[PaletteColor, ...] | None = None,
    partial_fit_max_renders: int | None = DEFAULT_PARTIAL_FIT_MAX_RENDERS,
    proposal_limit: int = PROPOSAL_LIMIT,
    dim_spread_threshold: float = DIM_SPREAD_THRESHOLD,
    light_cast: tuple[float, float, float] = LIGHT_CAST,
    light_cast_damping: float = LIGHT_CAST_DAMPING,
    light_cast_percentile: float = CAST_PERCENTILE,
    light_cast_region: int = CAST_REGION,
    propose_colour_gap: float = PROPOSE_COLOUR_GAP,
    propose_agreement: float = PROPOSE_AGREEMENT,
    depth_scale: float = DEPTH_SCALE,
    visibility_power: float = VISIBILITY_POWER,
    disagreement_risk: float = DISAGREEMENT_RISK,
    obliquity_power: float = OBLIQUITY_POWER,
    reconcile_exposure: bool = RECONCILE_EXPOSURE,
    face_shrink: float = FACE_SHRINK,
    max_face_variation: float = MAX_FACE_VARIATION,
    uniformity_scale: float = UNIFORMITY_SCALE,
    texture_weight: float = TEXTURE_WEIGHT,
    occlusion_strength: float = OCCLUSION_STRENGTH,
    simplicity_weight: float = SIMPLICITY_WEIGHT,
    shape_weight: float = SHAPE_WEIGHT,
    palette_support: float = PALETTE_SUPPORT,
    palette_prune_rounds: int = 2,
    palette_minimum: int = PALETTE_MINIMUM,
    local_support_region: int = LOCAL_SUPPORT_REGION,
    local_support_keep: int = LOCAL_SUPPORT_KEEP,
    local_support_wins: float = LOCAL_SUPPORT_WINS,
    margin_support: bool = MARGIN_SUPPORT,
    fallback_weight: float = FALLBACK_WEIGHT,
    support_weight: float = SUPPORT_WEIGHT,
    texture_resolution: bool = True,
    fit_tolerance: float = FIT_TOLERANCE,
    simulate_lighting: bool = True,
    lighting_rounds: int = 2,
    minimum_light_effect: float = MINIMUM_LIGHT_EFFECT,
    extra_emitters: dict[BlockCoordinate, int] | None = None,
    self_calibrate: bool = True,
    calibration_rounds: int = 3,
    slot_matching: bool = True,
    outlier_ceiling: float = OUTLIER_CEILING,
    weight_views: bool = True,
    view_tolerance: float = VIEW_TOLERANCE,
    family_priors: dict[str, float] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    cancel_event: Event | None = None,
    diagnostics: dict[str, object] | None = None,
    usable_masks: Mapping[str, np.ndarray] | None = None,
) -> BlockVolume:
    """Recover visible scene-grid blocks by calibrated projection and z-buffering."""
    if sample_radius < 0:
        raise ValueError("sample_radius cannot be negative")
    if min_views <= 0:
        raise ValueError("min_views must be positive")
    if max_color_distance is not None and max_color_distance < 0:
        raise ValueError("max_color_distance cannot be negative")
    if min_palette_margin < 0:
        raise ValueError("min_palette_margin cannot be negative")
    if not 0.0 <= min_state_agreement <= 1.0:
        raise ValueError("min_state_agreement must be between 0 and 1")
    if min_surface_contrast < 0:
        raise ValueError("min_surface_contrast cannot be negative")
    if adjacency_strength < 0.0:
        raise ValueError("adjacency_strength cannot be negative")
    if not 1 <= adjacency_min_neighbors <= 6:
        raise ValueError("adjacency_min_neighbors must be between 1 and 6")
    if (columns is not None and columns <= 0) or (rows is not None and rows <= 0):
        raise ValueError("columns and rows must be positive when provided")
    if family_priors is None:
        family_priors = DEFAULT_FAMILY_PRIORS
    palette_families = {item.state: item.family for item in palette}
    face_colors = _palette_face_colors(palette)
    merged: dict[BlockCoordinate, tuple[BlockState, float, set[str], list[BlockState], dict[BlockState, float]]] = {}
    total_frames = len(frames)
    coordinates = [
        (x, y, z)
        for x in range(scene.size[0])
        for y in range(scene.size[1])
        for z in range(scene.size[2])
    ]
    world_points = [_block_center(scene.to_world(coordinate)) for coordinate in coordinates]

    # Pass one: depth-buffer block centres to learn which blocks each frame sees.
    # This fixes the reconstruction's coordinates, so it stays as it was.
    per_frame_front: dict[str, dict[BlockCoordinate, float]] = {}
    for frame_index, (frame_id, image) in enumerate(frames.items(), start=1):
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("reconstruction cancelled")
        camera = cameras[frame_id]
        height, width = image.shape[:2]
        projections = project_points(world_points, camera, width=width, height=height)
        front: dict[tuple[int, int], tuple[float, BlockCoordinate]] = {}
        total_coordinates = len(coordinates)
        for coordinate_index, (coordinate, projection) in enumerate(zip(coordinates, projections), start=1):
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("reconstruction cancelled")
            if projection is None:
                continue
            pixel_x, pixel_y, depth = projection
            pixel = (round(pixel_x), round(pixel_y))
            if not (
                sample_radius <= pixel[0] < width - sample_radius
                and sample_radius <= pixel[1] < height - sample_radius
            ):
                continue
            previous = front.get(pixel)
            if previous is None or depth < previous[0]:
                front[pixel] = (depth, coordinate)
            if progress_callback is not None and coordinate_index % max(1, total_coordinates // 100) == 0:
                progress_callback(
                    (frame_index - 1) * total_coordinates + coordinate_index,
                    total_frames * total_coordinates,
                )
        per_frame_front[frame_id] = {
            coordinate: (depth, pixel) for pixel, (depth, coordinate) in front.items()
        }
        if progress_callback is not None:
            progress_callback(frame_index * total_coordinates, total_frames * total_coordinates)

    # The centre-pixel buffer marks a block as seen wherever it is the nearest
    # candidate, which in a scene with more pixels than blocks is nearly all of
    # them. Pruning the enclosed ones first leaves the actual surface, which is
    # the geometry the face sampler needs to reason about occlusion.
    seen_everywhere = {coordinate for seen in per_frame_front.values() for coordinate in seen}
    surface = _surface_of(seen_everywhere)
    proposed: set[BlockCoordinate] = set()
    stereo_cells: set[BlockCoordinate] = set()
    if propose_colour_gap > 0.0:
        # The pass above collapses to the region's bounding shell, so anything
        # standing inside a room is never proposed - including the light sources
        # whose absence stops the lighting simulation from running at all. Ask the
        # cameras directly which cells they agree on.
        from .occupancy import proposed_cells

        proposed = proposed_cells(
            scene, cameras, frames,
            colour_gap=propose_colour_gap, agreement=propose_agreement,
            gui_masks={name: create_gui_mask(image.shape[0], image.shape[1])
                       for name, image in frames.items()},
            minimum_area=minimum_face_area,
        )
        proposed = proposed - surface
        if diagnostics is not None:
            diagnostics["proposed_cells"] = len(proposed)
            diagnostics["proposed_cells_new"] = len(proposed)
    if extra_proposals:
        # Cells worked out by the caller - the residual pass renders this volume
        # against the photographs and proposes where they disagree, which needs a
        # first reconstruction to exist before it can run.
        outside = {cell for cell in extra_proposals if cell not in surface} - proposed
        proposed = proposed | outside
        stereo_cells = stereo_cells | outside
        if diagnostics is not None:
            diagnostics["extra_proposals_new"] = len(outside)
    if stereo_interior:
        # `proposed_cells` and the pass above both judge a cell by how alike its
        # views look, which on a repetitive build cannot tell a phantom from the
        # wall behind it - every version of that test put interior proposals at
        # 8-23% correct. Stereo asks instead which depth along a ray wins, and
        # only believes rays where the winner is unambiguous, which measured
        # 75-98%.
        from .stereo import stereo_surface

        found = stereo_surface(
            scene, cameras, frames,
            margin=stereo_margin, scale=stereo_scale,
        )
        interior = found - surface - proposed
        proposed = proposed | interior
        stereo_cells = interior
        if diagnostics is not None:
            diagnostics["stereo_cells"] = len(found)
            diagnostics["stereo_cells_new"] = len(interior)

    # Pass two: rectify each visible face and record what it looks like. Costs
    # wait until the per-frame colour cast has been estimated from these.
    raw: list[tuple[str, BlockCoordinate, np.ndarray, tuple[float, ...], str, float, np.ndarray | None]] = []
    face_rejections: dict[str, int] = {}
    calibrations: dict[str, AppearanceCalibration | None] = {}
    for frame_id, image in frames.items():
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("reconstruction cancelled")
        camera = cameras[frame_id]
        calibrations[frame_id] = (
            appearance_calibrations.get(frame_id)
            if appearance_calibrations
            else appearance_calibration
        )
        gui_mask = create_gui_mask(image.shape[0], image.shape[1])
        found = list(sample_visible_faces(
            surface, camera, scene, image, gui_mask=gui_mask,
            minimum_area=minimum_face_area, shrink=face_shrink,
            minimum_visible_fraction=minimum_visible_fraction,
            rejections=face_rejections, depth_scale=depth_scale,
        ))
        if proposed:
            # A proposed cell is a sampling target, not an occluder. Letting the
            # proposals into the set above would hide the walls behind them, which
            # cost more than the structure they recover was worth.
            # Keeping every face this returns, not just the proposals', is
            # deliberate: the surface faces come back under different occlusion
            # and that second opinion is load-bearing. Filtering to the proposals
            # alone measured 32.69% -> 28.98% on Template 6.
            found.extend(sample_visible_faces(
                proposed | surface, camera, scene, image, gui_mask=gui_mask,
                minimum_area=minimum_face_area, shrink=face_shrink,
                minimum_visible_fraction=minimum_visible_fraction,
                depth_scale=depth_scale,
            ))
        seen_here: set[tuple[BlockCoordinate, str]] = set()
        for sample in found:
            if (sample.coordinate, sample.slot) in seen_here:
                continue
            seen_here.add((sample.coordinate, sample.slot))
            pixels = sample.pixels
            if float(pixels.std()) < min_surface_contrast:
                face_rejections["low_contrast"] = face_rejections.get("low_contrast", 0) + 1
                continue
            # A patch that straddles an occluder is not a sample of one face.
            # Carpets, stairs and slabs sitting on a block are not in the
            # reconstruction's own geometry, so they cannot be culled; they can
            # still be recognised, because half the patch stops matching the
            # other half.
            observed_patch = sample.observed_patch
            uniformity = _face_uniformity(observed_patch, sample.coverage)
            if uniformity > max_face_variation:
                face_rejections["uniformity"] = face_rejections.get("uniformity", 0) + 1
                continue
            # A face whose quadrants agree is a cleaner read of one material than
            # a bigger face that is partly something else, so trust follows
            # internal consistency as well as size.
            # How much of the frame a face fills is how much of it was actually
            # measured. A face caught in a few dozen pixels is mostly its own
            # edges, and its neighbours bleed into the average; the same face
            # seen close up is the material itself. Raising the exponent lets a
            # clean, close view outvote a distant smear of the same block
            # without discarding the distant one, which for a face no other
            # frame caught is the only evidence there is.
            weight = float(sample.area**area_power) / (1.0 + sample.depth / 16.0)
            # A face the camera only half sees is half a measurement of it, and
            # the other half is whatever stands in front. Now that the visible
            # share is rasterised per pixel it is accurate enough to weigh with,
            # which grades that evidence instead of passing or failing it at a
            # threshold.
            if visibility_power:
                weight *= float(max(sample.visible, 0.0) ** visibility_power)
            # An edge-on face is a sliver of pixels stretched back into a square,
            # so it reports mostly its own resampling. A camera looking straight
            # at a face measured it; one grazing past it barely did.
            if obliquity_power:
                weight *= float(max(sample.facing, 0.0) ** obliquity_power)
            weight /= 1.0 + uniformity / max(uniformity_scale, 1e-6)
            raw.append((
                frame_id, sample.coordinate, sample.rgb,
                _patch_descriptor(observed_patch, sample.coverage), sample.slot, weight,
                _normalised_pattern(observed_patch, sample.coverage), sample.area,
            ))

    # Pass three: blocks whose faces were all too small, clipped or hidden behind
    # the HUD still deserve an answer, so fall back to their centre patch.
    covered = {coordinate for _f, coordinate, _c, _d, _s, _w, _p, _a in raw}
    for frame_id, image in frames.items():
        for coordinate, (depth, pixel) in per_frame_front[frame_id].items():
            if coordinate in covered:
                continue
            x, y = pixel
            patch = image[y - sample_radius : y + sample_radius + 1, x - sample_radius : x + sample_radius + 1]
            pixels = patch.astype(np.float32).reshape(-1, 3)
            if pixels.size == 0 or float(pixels.std()) < min_surface_contrast:
                continue
            # The centre patch is not a rectified face, so it carries no pattern.
            raw.append((
                frame_id, coordinate, np.median(pixels, axis=0), _patch_descriptor(patch),
                _dominant_face_slot(coordinate, cameras[frame_id], scene),
                fallback_weight / (1.0 + depth / 16.0), None, 0.0,
            ))

    # How trustworthy each block's evidence is, by where it came from. Two kinds
    # of face lie about their block. One points upward, and carpets, slabs and
    # stairs rest on top of blocks - geometry too thin for the reconstruction to
    # hold, so it can neither be measured nor culled - which covers upward faces
    # at twenty-odd percent against a few percent of every other direction. The
    # other is the centre-patch fallback, which is taken for blocks no face
    # sampling reached: those are the blocks nothing can actually see, and the
    # pixel at their centre belongs to whatever stands in front of them. Both
    # kinds are still worth keeping, because a weak reading of a block beats
    # none, but a block resting its case on them should follow its neighbours.
    upward_weight: dict[BlockCoordinate, float] = {}
    fallback_weight: dict[BlockCoordinate, float] = {}
    total_weight: dict[BlockCoordinate, float] = {}
    for _f, coordinate, _c, _d, slot, weight, _p, area in raw:
        total_weight[coordinate] = total_weight.get(coordinate, 0.0) + weight
        if area <= 0.0:
            fallback_weight[coordinate] = fallback_weight.get(coordinate, 0.0) + weight
        elif slot == "up":
            upward_weight[coordinate] = upward_weight.get(coordinate, 0.0) + weight
    upward_share = {
        coordinate: upward_weight.get(coordinate, 0.0) / total
        for coordinate, total in total_weight.items()
        if total > 0.0
    }
    fallback_share = {
        coordinate: fallback_weight.get(coordinate, 0.0) / total
        for coordinate, total in total_weight.items()
        if total > 0.0
    }

    if diagnostics is not None:
        resolved_faces = sum(1 for _f, _c, _r, _d, _s, _w, _p, area in raw if area > 0.0)
        fallback_faces = len(raw) - resolved_faces
        diagnostics["face_rejections"] = dict(sorted(face_rejections.items()))
        diagnostics["resolved_face_samples"] = resolved_faces
        diagnostics["fallback_samples"] = fallback_faces
        diagnostics["blocks_with_resolved_face"] = len(
            {c for _f, c, _r, _d, _s, _w, _p, area in raw if area > 0.0}
        )
        diagnostics["blocks_from_fallback_only"] = len(
            {c for _f, c, _r, _d, _s, _w, _p, area in raw if area <= 0.0}
            - {c for _f, c, _r, _d, _s, _w, _p, area in raw if area > 0.0}
        )
        diagnostics["mean_upward_evidence_share"] = (
            float(np.mean(list(upward_share.values()))) if upward_share else 0.0
        )
        diagnostics["coords_with_any_observation"] = len({c for _f, c, *_r in raw})
        diagnostics["blocks_seen_only_from_above"] = sum(
            1 for value in upward_share.values() if value > 0.999
        )

    if correct_light_cast:
        # Undo the light before anything else looks at the faces. The gate is the
        # brightness spread within a colour, the one measure of dimness that does
        # not ask the palette - and so the one that is not fooled by a palette
        # holding a plausible dark block for every material present.
        #
        # Ordering matters: `_self_calibrate` below fits a per-channel white
        # balance per frame from its own most confident faces. Run after this, it
        # fits the corrected appearance and the two compose; run before it, it
        # fits the dimmed one and the corrections fight, which measured as no
        # change at all on Template 6.
        spread = _scene_spread(raw)
        if diagnostics is not None:
            diagnostics["scene_spread"] = round(spread, 3)
        if spread >= dim_spread_threshold:
            raw = _uncast(raw, light_cast, damping=light_cast_damping,
                          percentile=light_cast_percentile,
                          region=light_cast_region)
            # One un-lighting at a time. Both `scene_gain` below and
            # `_self_calibrate`'s per-frame white balance estimate the same
            # quantity from a different angle, and stacking either on this one
            # over-corrects: Template 6 measured 30.61% at the shipped defaults,
            # 26.48% with this correction on top of self-calibration, and 30.21%
            # with self-calibration out of the way. Whichever runs, runs alone.
            dim_scene_threshold = 0.0
            self_calibrate = False
            if diagnostics is not None:
                diagnostics["light_cast_applied"] = True

    if reconcile_exposure:
        # Proposals are left in this fit. Keeping the stereo cells out of it
        # changes nothing at all (Template 6, 25.64% either way), while switching
        # the fit off entirely recovers 2.2pp of stereo's cost - so what harms the
        # stereo path here is the fit reacting to a changed observation set, not
        # the proposals' own contribution to it.
        exposure = _reconcile_exposure(raw)
        if diagnostics is not None:
            diagnostics["frame_exposure_gains"] = {
                frame_id: round(gain, 4) for frame_id, gain in sorted(exposure.items())
            }
        if any(abs(gain - 1.0) > 1e-3 for gain in exposure.values()):
            raw = [
                (
                    frame_id, coordinate, rgb * exposure.get(frame_id, 1.0), descriptor,
                    slot, weight, pattern, area,
                )
                for frame_id, coordinate, rgb, descriptor, slot, weight, pattern, area in raw
            ]

    if self_calibrate and not appearance_calibrations and appearance_calibration is None:
        # Proposals are deliberately left in. Keeping them out of this fit, by the
        # same argument that keeps them out of palette pruning, measured Template 6
        # at 12.22% against 32.69% - the per-frame white balance wants every face
        # it can get, and the uncertain ones still carry a frame's colour cast.
        calibrations = _self_calibrate(
            raw, PaletteIndex(palette, face_colors, family_priors), face_colors,
            outlier_ceiling, rounds=calibration_rounds, fit_exposure=fit_exposure,
        )

    index = PaletteIndex(palette, face_colors, family_priors, texture_weight)
    # Kept before pruning: judging how dark a scene is needs the whole
    # palette. Against the handful of candidates support pruning leaves,
    # every face matches something of about the right brightness and the
    # measurement collapses to one whatever the room is doing.
    unpruned_index = index

    occlusion: dict[tuple[BlockCoordinate, str], float] = {}
    if occlusion_strength > 0.0:
        for _frame_id, coordinate, _rgb, _descriptor, slot, _weight, _pattern, _area in raw:
            key = (coordinate, slot)
            if key not in occlusion:
                shade = ambient_occlusion(coordinate, slot, surface)
                occlusion[key] = 1.0 - occlusion_strength * (1.0 - shade)

    slot_factors: dict[str, float] = {}
    scene_gain = 1.0

    def score(field: LightField | None) -> list[tuple[str, BlockCoordinate, np.ndarray, float]]:
        scored = []
        for frame_id, coordinate, rgb, descriptor, slot, weight, pattern, area in raw:
            factor = field.face_factor(coordinate, slot) if field is not None else 1.0
            factor *= occlusion.get((coordinate, slot), 1.0)
            if scene_gain < 1.0:
                factor *= scene_gain
            if slot_factors:
                # normalised against the scene, so this expresses only the
                # difference between facings and not the overall exposure, which
                # the field already carries
                factor *= slot_factors.get(slot, 1.0)
            # A face only a few pixels across cannot show its grain, so trust the
            # pattern in proportion to how much of it the screen actually
            # resolved: one screen pixel per texel is a full vote.
            resolved = min(1.0, area / (PATCH_SIZE * PATCH_SIZE)) if texture_resolution else 1.0
            costs, best = index.costs(
                rgb, descriptor, slot, calibration=calibrations.get(frame_id), light=factor,
                pattern=pattern, texture_weight=texture_weight * resolved,
                outlier_ceiling=outlier_ceiling, simplicity_weight=simplicity_weight,
                shape_weight=shape_weight,
            )
            if fit_tolerance > 0.0:
                # How well the best candidate explains a face says whether the
                # face is worth listening to. A patch that caught an occluder or
                # straddled an edge matches nothing in the palette particularly
                # well, and averaging it in corrupts a block that other views saw
                # cleanly.
                weight = weight * float(np.exp(-best / fit_tolerance))
            scored.append((frame_id, coordinate, costs, weight))
        return scored

    observations = score(None)
    for _round in range(palette_prune_rounds if palette_support > 0.0 else 0):
        # A build is made of a handful of materials, but the palette offers every
        # full cube in the game. Most of those never win anything except a few
        # stray faces they happen to sit closest to, and each one they steal is a
        # block wrong. Keeping only the candidates the first pass actually found
        # support for removes that tail without deciding in advance what a scene
        # is allowed to contain.
        # Stereo cells are the least certain in the scene, so they get a material
        # but no vote on which materials exist. Pruning keeps only a
        # handful of candidates - seven to eleven on Template 6 - and a few wrong
        # votes there remove a material the build actually uses, which no later
        # stage can put back. Letting 31 stereo proposals into this decision took
        # Template 6 from 33.46% to 3.40%; with pruning disabled the same
        # proposals cost 0.78pp, which is how the interaction was found.
        # Once stereo is contributing, *no* proposal votes - not the stereo cells
        # and not `proposed_cells` either. Excluding only the stereo ones leaves
        # Template 6 at 3.40%; excluding every proposal recovers it to 25.64%, so
        # the older proposals' votes are only harmful in stereo's company. With
        # stereo off nothing changes, which keeps the shipped path exactly as
        # measured: excluding proposals unconditionally costs it 0.77pp.
        deciding = (
            [entry for entry in observations if entry[1] not in proposed]
            if stereo_cells else observations
        )
        keep = _supported(deciding or observations, index, palette_support,
                          palette_minimum, margin_support)
        if local_support_keep > 0:
            nearby = _locally_supported(
                deciding or observations, index,
                region=local_support_region, keep=local_support_keep,
                minimum_wins=local_support_wins,
            )
            if nearby:
                keep = np.array(sorted(set(int(i) for i in keep) | nearby), dtype=np.intp)
        if keep.size < index.count:
            index = index.restricted(keep)
            observations = score(None)
        if support_weight > 0.0:
            # How much of the scene a candidate already explains is evidence
            # about how likely it is anywhere else in it. Builds reuse a few
            # materials, so a candidate winning a tenth of the faces is a better
            # prior guess than one winning a hundredth.
            index = index.with_prior(_support_prior(observations, index, support_weight))
            observations = score(None)
    if simulate_lighting:
        # Classify once unlit, let the blocks that came out say where the light
        # is, then classify again against colours dimmed the way that light would
        # dim them. A block one level darker is a different material otherwise.
        field: LightField | None = None
        # The surface a face was genuinely resolved on: what really stops light.
        lit_solid = {
            coordinate
            for _f, coordinate, _r, _d, _s, _w, _p, area in raw
            if area > 0.0
        }
        found_emitters = (
            _detect_emitters(raw, palette, face_colors) if detect_emitters else {}
        )
        if found_emitters:
            merged_emitters = dict(extra_emitters or {})
            for coordinate, level in found_emitters.items():
                merged_emitters[coordinate] = max(merged_emitters.get(coordinate, 0), level)
            extra_emitters = merged_emitters
        if diagnostics is not None:
            diagnostics["detected_emitters"] = len(found_emitters)
        for _round in range(max(1, lighting_rounds)):
            decided = _decide(
                observations, index, adjacency_strength, adjacency_min_neighbors,
                bounded_adjacency, evidence_power,
                _coupling_boost(observations, index, upward_share, fallback_share,
                                upward_risk, fallback_risk, disagreement_risk),
                adopt_boost=adopt_boost,
            )
            if not decided:
                break
            if dim_scene_threshold > 0.0:
                brightness_now = _scene_brightness(
                    raw, decided, face_colors, unpruned_index,
                    texture_weight, outlier_ceiling, simplicity_weight,
                    applied=scene_gain,
                )
                if diagnostics is not None:
                    diagnostics["scene_brightness"] = round(brightness_now, 3)
                if brightness_now >= dim_scene_threshold:
                    break  # already as bright as the jar; there is nothing to undo
                if global_exposure:
                    # How far under the jar the whole scene sits, measured on the
                    # faces the classifier chose for itself. A light field can only
                    # express this where it has found sources; this expresses it
                    # everywhere, which is what a uniformly dim room needs.
                    #
                    # Compounded across rounds rather than replaced. The estimate
                    # is biased toward one - it is taken from faces that already
                    # matched well - so a single pass under-corrects, and measuring
                    # the same untouched pixels again would return the same answer
                    # for ever. Measuring what is left after the correction so far
                    # lets it converge.
                    scene_gain = max(
                        scene_gain * brightness_now, MINIMUM_LIGHT_FACTOR
                    )
                    if diagnostics is not None:
                        diagnostics["scene_gain"] = round(scene_gain, 3)
            sources = dict(extra_emitters or {})
            if trilaterate_sources:
                # Where a light source has to be, given how bright the scene is
                # around it - whether or not any camera can see it. Builders hide
                # emitters under carpets and slabs, so a field built only from
                # sources the classifier recognised reports darkness in a room
                # that is plainly lit.
                measured = [
                    (coordinate, slot, rgb, face_colors[decided[coordinate]][slot])
                    for _f, coordinate, rgb, _d, slot, _w, _p, area in raw
                    if area > 0.0
                    and coordinate in decided
                    and slot in face_colors.get(decided[coordinate], {})
                ]
                # Occupancy here must be the surface actually resolved, not every
                # cell the classifier had an opinion about. The fallback gives an
                # opinion on nearly the whole region, and treating that as solid
                # leaves no air for light to travel through and no air cell for a
                # face to be lit by.
                levels = estimate_light_levels(measured, lit_solid, scene.size)
                located = locate_sources(levels, scene.size, lit_solid,
                                         keep=source_candidates)
                for coordinate, level in located.items():
                    sources[coordinate] = max(sources.get(coordinate, 0), level)
                if diagnostics is not None:
                    diagnostics["trilaterated_sources"] = len(located)
                    diagnostics["lit_cells_measured"] = len(levels)
            # Only the surface actually resolved blocks light. Handing over every
            # cell the classifier had an opinion about makes the whole region
            # opaque, light stops dead at each source, and the field comes back
            # level 0 almost everywhere - which the un-lighting then turns into a
            # flat brightening by the clamp rather than a spatial correction.
            blocking = {
                coordinate: state
                for coordinate, state in decided.items()
                if coordinate in lit_solid
            }
            trial = light_field(scene.size, blocking, extra_emitters=sources)
            if int(trial.levels.max()) == 0:
                break  # nothing found that emits light
            if vanilla_light_response:
                # With the client at default gamma and no shaders, how a light
                # level maps to a pixel is not an unknown to be fitted - it is
                # Minecraft's own curve. Fitting it instead needs blocks that are
                # already classified correctly, which is exactly what a dim scene
                # does not have, so the fit collapses to flat and the field then
                # dims nothing.
                response = LightResponse(floor=0.0, scale=1.0, confidence=1.0)
            else:
                response = _measure_response(raw, decided, face_colors, trial)
                # Only correct for light the simulation can actually account for.
                if response.confidence < minimum_light_effect:
                    break
            candidate_field = trial.with_response(response)
            if ambient_levels:
                # Light the simulation cannot account for still reaches a room -
                # from sources outside the crop, from the ones trilateration
                # missed, from the sky. Without a floor those cells sit at level
                # zero and the un-lighting brightens them by the clamp, which
                # wrecks the fit everywhere the found sources do not reach. The
                # floor is chosen by which value explains the faces best; the
                # field's shape is already fixed by the sources, so this is one
                # number rather than a free field.
                best_cost, best_field = np.inf, candidate_field
                for ambient in ambient_levels:
                    lifted = LightField(
                        np.maximum(trial.levels, int(ambient)), response
                    )
                    cost = _explanatory_cost(raw, index, lifted, calibrations,
                                             texture_weight, outlier_ceiling,
                                             simplicity_weight)
                    if cost < best_cost:
                        best_cost, best_field = cost, lifted
                        if diagnostics is not None:
                            diagnostics["ambient_level"] = int(ambient)
                candidate_field = best_field
            if light_must_help:
                lit = _explanatory_cost(raw, index, candidate_field, calibrations,
                                        texture_weight, outlier_ceiling, simplicity_weight)
                unlit = _explanatory_cost(raw, index, None, calibrations,
                                          texture_weight, outlier_ceiling, simplicity_weight)
                if diagnostics is not None:
                    diagnostics["light_cost_unlit"] = round(unlit, 3)
                    diagnostics["light_cost_lit"] = round(lit, 3)
                if lit >= unlit:
                    break  # the scene is already as bright as the jar
            field = candidate_field
            if slot_response:
                # The field gives a cell one level and every face of it shares
                # that level, so a directional bias survives it. Measured the same
                # way the scene brightness is - on faces the classifier chose for
                # itself, where the leftover brightness is honest.
                slot_factors = _slot_brightness(
                    raw, decided, face_colors, unpruned_index,
                    texture_weight, outlier_ceiling, simplicity_weight,
                )
                if diagnostics is not None:
                    diagnostics["slot_brightness"] = {
                        slot: round(value, 3) for slot, value in sorted(slot_factors.items())
                    }
            observations = score(field)
        lighting_field = field
    else:
        lighting_field = None

    view_weights = _view_weights(observations, index, view_tolerance) if weight_views else {}
    totals, weights = _accumulate_costs(observations, view_weights)
    observed_frames: dict[BlockCoordinate, set[str]] = {}
    for frame_id, coordinate, _costs, weight in observations:
        if weight * view_weights.get(frame_id, 1.0) > 0.0:
            observed_frames.setdefault(coordinate, set()).add(frame_id)

    for coordinate, total in totals.items():
        mean = total / max(weights[coordinate], 1e-6)
        distance = float(mean.min())
        if max_color_distance is not None and distance > max_color_distance:
            continue
        if mean.size > 1:
            lowest = np.partition(mean, 1)[:2]
            margin = float(lowest.max() - lowest.min())
        else:
            margin = distance
        if margin < min_palette_margin:
            continue
        confidence = max(0.0, min(1.0, 1.0 - distance / 441.67))
        candidate_scores = _top_candidates(mean, index)
        if not candidate_scores:
            continue
        state = max(candidate_scores, key=candidate_scores.get)
        merged[coordinate] = (
            state,
            confidence,
            observed_frames.get(coordinate, set()),
            [state],
            candidate_scores,
        )

    volume = BlockVolume()
    if adjacency_strength:
        _smooth_adjacent_states(merged, adjacency_strength, adjacency_min_neighbors,
                                bounded=bounded_adjacency, evidence_power=evidence_power,
                                coupling_boost=_coupling_boost(
                                    observations, index, upward_share, fallback_share,
                                    upward_risk, fallback_risk, disagreement_risk),
                                adopt_boost=adopt_boost)
    if diagnostics is not None:
        diagnostics["merged_coordinates"] = len(merged)
    if require_resolved_face:
        # Only place a block where a face was actually resolved. The centre-patch
        # fallback reaches nearly every cell in the region, so without this the
        # volume is filled solid and `_remove_enclosed_blocks` then hollows it back
        # out to the region's bounding shell - which is how a geometry stage that
        # never worked produced plausible-looking rooms and lost everything
        # standing inside them.
        resolved = {c for _f, c, _r, _d, _s, _w, _p, area in raw if area > 0.0}
        merged = {c: value for c, value in merged.items() if c in resolved}
        if diagnostics is not None:
            diagnostics["merged_after_resolved_filter"] = len(merged)
    for coordinate, (state, confidence, frame_ids, states, candidate_scores) in merged.items():
        state = _choose_state_with_local_family_support(coordinate, candidate_scores, merged, palette_families)
        if state is None:
            continue
        state_count = sum(candidate == state for candidate in states)
        if state_count / len(states) < min_state_agreement:
            continue
        if len(frame_ids) < min_views:
            continue
        volume.set(coordinate, BlockObservation(state=state, confidence=confidence, frame_ids=sorted(frame_ids)))
    if surface_only:
        # Pruning enclosed blocks is what turns the filled volume into a surface.
        # It also deletes anything standing inside a room, because the fallback has
        # filled the room around it - so the cells the cameras positively agreed on
        # are restored afterwards rather than being pruned along with the filling.
        restore = {
            coordinate: merged[coordinate]
            for coordinate in proposed
            if coordinate in merged
        }
        _remove_enclosed_blocks(volume)
        restore = {c: v for c, v in restore.items() if c not in volume.blocks}
        for coordinate, (state, confidence, frame_ids, states, scores) in restore.items():
            chosen = _choose_state_with_local_family_support(
                coordinate, scores, merged, palette_families
            )
            if chosen is None:
                continue
            volume.set(coordinate, BlockObservation(
                state=chosen, confidence=confidence, frame_ids=sorted(frame_ids)))
        if diagnostics is not None:
            diagnostics["restored_after_pruning"] = len(restore)

    if fit_partial_shapes:
        # The thin-block chain: render what will ship against the photographs,
        # propose the empty cell in front of every face they disagree about, and
        # fit a partial shape and material to each by rendering candidates in.
        #
        # It runs *here*, on the pruned volume, and not against `merged` earlier.
        # Before pruning the volume is filled solid, so the render is a box and
        # the residual describes that box: proposing against it found three cells
        # where the finished surface offers forty.
        from .residual import residual_proposals
        from .shape_fit import fit_states

        settled = {c: entry.state for c, entry in volume.blocks.items()}
        fitting = shape_palette if shape_palette is not None else palette
        wanted = residual_proposals(
            settled, scene, cameras, frames, fitting, limit=proposal_limit)
        fitted = fit_states(
            list(wanted), settled, scene, cameras, frames, fitting,
            usable_masks=usable_masks,
            diagnostics=diagnostics,
            max_candidate_renders=partial_fit_max_renders,
        )
        fit_support = getattr(fitted, "supporting_frames", {})
        for coordinate, state in fitted.items():
            if state is None:
                # ``None`` is an explicit, evidence-backed deletion from the
                # sparse volume; an absent fit entry still means "retain".
                volume.blocks.pop(coordinate, None)
                continue
            frame_ids = fit_support.get(coordinate, ())
            volume.set(coordinate, BlockObservation(
                state=state, confidence=0.0, frame_ids=list(frame_ids)))
        if diagnostics is not None:
            diagnostics["residual_proposals"] = len(wanted)
            diagnostics["shapes_fitted"] = len(fitted)
    return volume


def _surface_of(coordinates: set[BlockCoordinate]) -> set[BlockCoordinate]:
    """Drop coordinates whose six neighbours are all present."""
    directions = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    return {
        coordinate
        for coordinate in coordinates
        if not all(
            (coordinate[0] + dx, coordinate[1] + dy, coordinate[2] + dz) in coordinates
            for dx, dy, dz in directions
        )
    }


def _dominant_face_slot(
    coordinate: BlockCoordinate, camera: CameraModel, scene: SceneTransform
) -> str:
    """The world face slot most directly turned toward the camera."""
    center = np.asarray(_block_center(scene.to_world(coordinate)), dtype=float)
    eye = np.asarray(camera.position, dtype=float) + np.asarray([0.0, camera.eye_height, 0.0])
    direction = eye - center
    axis = int(np.argmax(np.abs(direction)))
    positive = direction[axis] > 0.0
    return (("east", "west"), ("up", "down"), ("south", "north"))[axis][0 if positive else 1]


def _face_uniformity(patch: np.ndarray, coverage: np.ndarray | None = None) -> float:
    """Largest appearance gap between the quadrants of a face patch.

    One block face is one material, so its quadrants should agree. A large gap
    means the patch caught something else: an occluding block the reconstruction
    does not know about, or a neighbour dragged in by a camera-pose error.
    """
    half = patch.shape[0] // 2
    if half < 1:
        return 0.0
    quadrants = []
    for y in (0, half):
        for x in (0, half):
            pixels = patch[y : y + half, x : x + half].reshape(-1, 3)
            weights = None if coverage is None else coverage[y : y + half, x : x + half].reshape(-1)
            if weights is not None and float(weights.sum()) > 1e-6:
                quadrants.append(np.average(pixels, axis=0, weights=weights))
            else:
                quadrants.append(np.mean(pixels, axis=0))
    return max(
        _appearance_distance(first, second)
        for index, first in enumerate(quadrants)
        for second in quadrants[index + 1 :]
    )


def _self_calibrate(
    raw: list[tuple[str, BlockCoordinate, np.ndarray, tuple[float, ...], str, float, np.ndarray | None]],
    index: PaletteIndex,
    face_colors: dict[BlockState, dict[str, np.ndarray]],
    outlier_ceiling: float,
    *,
    rounds: int = 3,
    sample_size: int = 250,
    minimum_cast: float = MINIMUM_COLOUR_CAST,
    fit_exposure: bool = False,
    exposure_floor: float = EXPOSURE_FLOOR,
    exposure_ceiling: float = EXPOSURE_CEILING,
) -> dict[str, AppearanceCalibration]:
    """Estimate each frame's colour cast from its own most confident faces.

    Screenshots of the same room differ in white balance - warm torch light in
    one, cooler sky light in another - and the shift is small in absolute terms
    but decisive between materials. A pale floor read under warm light lands
    exactly on a different, warmer wood.
    """
    identity = AppearanceCalibration((1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    frames = {frame_id for frame_id, *_rest in raw}
    calibrations = {frame_id: identity for frame_id in frames}
    if not raw:
        return calibrations
    # A white balance is one number per channel per frame, so it converges on a
    # few hundred faces. Fitting it against every observation, several rounds
    # over, was three quarters of a reconstruction's runtime for no precision.
    per_frame: dict[str, list] = {}
    for observation in raw:
        per_frame.setdefault(observation[0], []).append(observation)
    sample: list = []
    for observations in per_frame.values():
        step = max(1, len(observations) // max(1, sample_size))
        sample.extend(observations[::step])
    for _round in range(max(1, rounds)):
        matches: dict[str, list[tuple[float, np.ndarray, np.ndarray]]] = {f: [] for f in frames}
        for frame_id, _coordinate, rgb, descriptor, slot, _weight, _pattern, _area in sample:
            costs, _best = index.costs(
                rgb, descriptor, slot, calibration=calibrations[frame_id],
                outlier_ceiling=outlier_ceiling,
            )
            if costs.size == 0:
                continue
            winner = index.states[int(np.argmin(costs))]
            margin = float(np.partition(costs, 1)[1]) if costs.size > 1 else 0.0
            expected = face_colors[winner].get(slot)
            if expected is None or float(expected.min()) < 8.0 or float(rgb.min()) < 8.0:
                continue
            matches[frame_id].append((margin, np.asarray(rgb, dtype=np.float32), expected))
        for frame_id, found in matches.items():
            if len(found) < 12:
                continue
            found.sort(key=lambda item: -item[0])
            kept = found[: max(12, len(found) // 3)]
            ratios = np.stack([observed / expected for _m, observed, expected in kept])
            if fit_exposure:
                # A room lit below full brightness photographs darker than the
                # textures it was built from, and every block in it then lands on
                # whatever the palette offers that is genuinely darker - stone
                # reads as grey terracotta, calcite as something black. Where the
                # light sources are known the simulation undoes that, but a cave
                # lit from outside the crop has no emitters to find, and the
                # scene's own most confident matches are the only measure of how
                # far under the jar's brightness it was captured. Keeping the
                # magnitude of the fitted ratio instead of dividing it out lets
                # the estimate carry that exposure as well as the colour cast.
                scale = np.clip(np.median(ratios, axis=0), exposure_floor, exposure_ceiling)
            else:
                scale = np.clip(np.median(ratios, axis=0), 0.5, 2.0)
                scale = scale / float(np.mean(scale))
            # Only correct a cast the frame actually has. A balance fitted from a
            # sample always lands a little off identity, and applying that noise
            # moves block decisions for nothing.
            if float(np.abs(scale - 1.0).max()) < minimum_cast:
                calibrations[frame_id] = identity
                continue
            calibrations[frame_id] = AppearanceCalibration(
                tuple(float(value) for value in scale), (0.0, 0.0, 0.0)
            )
    return calibrations


def _palette_face_textures(
    palette: tuple[PaletteColor, ...],
) -> dict[str, tuple[list[BlockState], np.ndarray]]:
    """Per face slot, the candidate states and their normalised texture patterns.

    Each texture is reduced to zero-mean, unit-norm luminance, so a dot product
    against a patch prepared the same way is the correlation between them. That
    is invariant to how brightly the face happened to be lit, which is the one
    thing block lighting changes most and material identity changes least.
    """
    import base64
    from io import BytesIO

    per_slot: dict[str, tuple[list[BlockState], list[np.ndarray]]] = {}
    for item in palette:
        for slot, encoded in item.face_thumbnails:
            try:
                from PIL import Image

                pixels = np.asarray(
                    Image.open(BytesIO(base64.b64decode(encoded))), dtype=np.float32
                )
            except (ValueError, OSError, TypeError, ImportError):
                continue
            if pixels.ndim != 3 or pixels.shape[:2] != (PATCH_SIZE, PATCH_SIZE):
                continue
            vector = _normalised_pattern(pixels)
            if vector is None:
                continue
            states, vectors = per_slot.setdefault(slot, ([], []))
            states.append(item.state)
            vectors.append(vector)
    return {
        slot: (states, np.stack(vectors)) for slot, (states, vectors) in per_slot.items() if vectors
    }


def _normalised_pattern(patch: np.ndarray, coverage: np.ndarray | None = None) -> np.ndarray | None:
    """Zero-mean, unit-norm luminance of a face patch, or ``None`` if it is flat."""
    luminance = patch.mean(axis=2).reshape(-1)
    weights = None if coverage is None else coverage.reshape(-1)
    mean = float(np.average(luminance, weights=weights)) if weights is not None and float(weights.sum()) > 1e-6 else float(luminance.mean())
    centred = luminance - mean
    if weights is not None:
        centred = centred * np.sqrt(np.clip(weights, 0.0, 1.0))
    norm = float(np.linalg.norm(centred))
    if norm < 1e-3:
        return None
    return (centred / norm).astype(np.float32)


AXIS_OF_SLOT = {"east": 0, "west": 0, "up": 1, "down": 1, "south": 2, "north": 2}


def slot_coverage(item: PaletteColor, slot: str, tolerance: float = 1e-3) -> float:
    """How much of a cell face this block's shape hides, seen through that face.

    The quantity is the shape's **projected footprint** along the face's axis - the
    union of its boxes' cross sections - not whether it touches the cell wall.
    Wall contact was the first attempt and it is wrong in the case that matters: a
    carpet's top surface sits at y=0.0625, so it never touches the ceiling of its
    cell and scored zero on `up`, the very face a camera sees it through. With the
    maximum shape penalty on its only visible face, a carpet could never win a cell,
    and residual proposals that should have become carpets came back as
    `birch_planks`.

    By footprint a carpet seen from above covers all of it (1.0) and from the side a
    sixteenth; a bottom slab covers all of it from above and half from the side; a
    fence post a sixteenth from above and a quarter from the side. Direction does
    not matter, only the axis: a shape occludes the same cross section whichever
    way you look along it.

    This deliberately no longer separates a bottom slab from a full block seen from
    directly overhead - they really do hide the same area - so that call is left to
    the side faces and to depth, which is where the evidence actually is.
    """
    if item.full_cube or not item.shape_boxes:
        return 1.0
    axis = AXIS_OF_SLOT.get(slot)
    if axis is None:
        return 1.0
    others = [index for index in range(3) if index != axis]
    rectangles = [
        ((low[others[0]], low[others[1]]), (high[others[0]], high[others[1]]))
        for low, high in item.shape_boxes
    ]
    rectangles = [r for r in rectangles
                  if r[1][0] - r[0][0] > tolerance and r[1][1] - r[0][1] > tolerance]
    if not rectangles:
        return 0.0
    # Small boxes per model, so the union is cheapest over the grid their edges
    # induce rather than by a sweep.
    edges_u = sorted({v for r in rectangles for v in (r[0][0], r[1][0])})
    edges_v = sorted({v for r in rectangles for v in (r[0][1], r[1][1])})
    area = 0.0
    for u0, u1 in zip(edges_u, edges_u[1:]):
        for v0, v1 in zip(edges_v, edges_v[1:]):
            midpoint = ((u0 + u1) / 2.0, (v0 + v1) / 2.0)
            if any(r[0][0] <= midpoint[0] <= r[1][0] and r[0][1] <= midpoint[1] <= r[1][1]
                   for r in rectangles):
                area += (u1 - u0) * (v1 - v0)
    return min(1.0, max(0.0, area))


class PaletteIndex:
    """Palette laid out as arrays so one observation is scored in a few array ops.

    Scoring walked every candidate in Python, which at several thousand faces and
    several hundred candidates was tens of millions of three-element numpy calls
    and most of a reconstruction's runtime. Everything a candidate contributes -
    its colour per face slot, its texture descriptors, its family prior, its
    texture pattern - is stacked here once, so a face costs a handful of whole
    array operations instead.
    """

    def __init__(
        self,
        palette: tuple[PaletteColor, ...],
        face_colors: dict[BlockState, dict[str, np.ndarray]],
        family_priors: dict[str, float] | None,
        texture_weight: float = 0.0,
    ) -> None:
        self.states: list[BlockState] = [item.state for item in palette]
        self.count = len(self.states)
        self.colours = {
            slot: np.stack([face_colors[item.state][slot] for item in palette]).astype(np.float64)
            for slot in FACE_SLOTS
        }
        rows: list[tuple[float, ...]] = []
        owners: list[int] = []
        for index, item in enumerate(palette):
            for descriptor in item.descriptors:
                rows.append(descriptor)
                owners.append(index)
        self.descriptors = np.asarray(rows, dtype=np.float64) if rows else None
        self.descriptor_owner = np.asarray(owners, dtype=np.intp) if rows else None
        self.priors = (
            np.asarray([max(0.1, family_priors.get(item.family, 1.0)) for item in palette])
            if family_priors
            else None
        )
        # Among candidates that explain a face equally well, prefer the one whose
        # appearance is simplest: the number of distinct faces it wears. Plain
        # planks show one texture, a log two, a workstation three or more. A wall
        # seen only side-on cannot tell dark oak planks from a cartography table,
        # which carries the same wood on its sides and a map on top, and the
        # planks are the simpler explanation of what was actually seen.
        # Plain building material wears one face (planks, stone) or two (a log's
        # bark and end, a blackstone's top and side). Three or more distinct
        # faces means a fixture - a workstation, a furnace, an observer - whose
        # sides borrow an ordinary material. A wall seen only side-on cannot tell
        # dark oak planks from a cartography table wearing the same wood, and the
        # planks are the simpler explanation of what was actually seen. Counting
        # only faces beyond the second leaves the log-versus-planks call, which
        # real texture evidence settles, untouched.
        distinct = np.asarray(
            [len({tuple(colour) for _slot, colour in item.face_map}) or 1 for item in palette],
            dtype=np.float64,
        )
        self.simplicity = np.maximum(distinct - 2.0, 0.0)
        # How much of each cell face every candidate's shape actually fills, so a
        # partial block is not asked to explain a face it only partly occupies.
        self.coverage = {
            slot: np.array([slot_coverage(item, slot) for item in palette], dtype=np.float64)
            for slot in FACE_SLOTS
        }
        self.prior: np.ndarray | None = None
        self.textures: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if texture_weight > 0.0:
            position = {state: index for index, state in enumerate(self.states)}
            for slot, (states, matrix) in _palette_face_textures(palette).items():
                mask = np.zeros(self.count, dtype=bool)
                aligned = np.zeros((self.count, matrix.shape[1]), dtype=np.float32)
                for row, state in enumerate(states):
                    index = position.get(state)
                    if index is not None:
                        aligned[index] = matrix[row]
                        mask[index] = True
                self.textures[slot] = (aligned, mask)

    def with_prior(self, prior: np.ndarray) -> "PaletteIndex":
        """A view of this index carrying an additive per-candidate cost."""
        clone = object.__new__(PaletteIndex)
        clone.__dict__.update(getattr(self, "__dict__", {}))
        for name in ("states", "count", "colours", "descriptors", "descriptor_owner",
                     "priors", "textures", "simplicity", "prior"):
            setattr(clone, name, getattr(self, name, None))
        clone.prior = np.asarray(prior, dtype=np.float64)
        return clone

    def restricted(self, keep: np.ndarray) -> "PaletteIndex":
        """A view of this index holding only the given candidate indices."""
        clone = object.__new__(PaletteIndex)
        clone.states = [self.states[int(i)] for i in keep]
        clone.count = len(clone.states)
        clone.colours = {slot: value[keep] for slot, value in self.colours.items()}
        if self.descriptors is not None:
            position = {int(old): new for new, old in enumerate(keep)}
            rows, owners = [], []
            for row, owner in enumerate(self.descriptor_owner):
                moved = position.get(int(owner))
                if moved is not None:
                    rows.append(self.descriptors[row])
                    owners.append(moved)
            clone.descriptors = np.asarray(rows) if rows else None
            clone.descriptor_owner = np.asarray(owners, dtype=np.intp) if rows else None
        else:
            clone.descriptors = None
            clone.descriptor_owner = None
        clone.priors = None if self.priors is None else self.priors[keep]
        clone.simplicity = self.simplicity[keep]
        clone.coverage = {slot: values[keep] for slot, values in self.coverage.items()}
        clone.textures = {
            slot: (aligned[keep], mask[keep]) for slot, (aligned, mask) in self.textures.items()
        }
        return clone

    def costs(
        self,
        rgb: np.ndarray,
        descriptor: tuple[float, ...],
        slot: str,
        *,
        calibration: AppearanceCalibration | None = None,
        light: float = 1.0,
        pattern: np.ndarray | None = None,
        texture_weight: float = 0.0,
        outlier_ceiling: float = 24.0,
        simplicity_weight: float = 0.0,
        shape_weight: float = 0.0,
    ) -> tuple[np.ndarray, float]:
        """Relative, capped cost of every candidate for one observed face."""
        # Undo the lighting rather than dim every candidate to match it. Dividing
        # the observation back up to how the block would look at full brightness
        # keeps every face on the same footing as the jar's own textures: a face
        # in shadow then has the same room to be wrong as a lit one, where
        # dimming the expectation squeezes the candidates together and makes a
        # dark face weak evidence for anything.
        if light < 1.0:
            rgb = rgb / max(light, MINIMUM_LIGHT_FACTOR)
        rgb = np.asarray(rgb, dtype=np.float64)
        expected = self.colours[slot]
        if calibration is not None:
            expected = expected * np.asarray(calibration.scale) + np.asarray(calibration.offset)
        difference = rgb - expected
        straight = np.sqrt(np.einsum("ij,ij->i", difference, difference))
        observed_total = max(float(rgb.sum()), 1e-6)
        expected_total = np.maximum(expected.sum(axis=1, keepdims=True), 1e-6)
        chroma = rgb / observed_total - expected / expected_total
        chroma = np.sqrt(np.einsum("ij,ij->i", chroma, chroma)) * 441.67
        distance = (1.0 - CHROMA_WEIGHT) * straight + CHROMA_WEIGHT * chroma
        if self.descriptors is not None and descriptor:
            gap = self.descriptors - np.asarray(descriptor, dtype=np.float64)
            gap = np.sqrt(np.einsum("ij,ij->i", gap, gap))
            closest = np.full(self.count, np.inf)
            np.minimum.at(closest, self.descriptor_owner, gap)
            distance = distance + 80.0 * np.where(np.isfinite(closest), closest, 0.0)
        if self.priors is not None:
            distance = distance / self.priors
        if pattern is not None and texture_weight > 0.0:
            entry = self.textures.get(slot)
            if entry is not None:
                aligned, mask = entry
                correlation = aligned[mask] @ pattern
                if correlation.size:
                    # Score each candidate against how well a typical candidate
                    # explains this patch, not against a perfect match. Real
                    # screenshots are far blurrier than the 16-pixel source
                    # textures, so every correlation is small and an absolute
                    # penalty charges nearly the full weight to everyone -
                    # including, through the fixed fallback, the candidates with
                    # no pattern for this slot, which were meant to be neutral.
                    # Centring makes the term zero-sum: it only ever expresses
                    # which candidate matches better than the rest, and a patch
                    # too blurred to separate them contributes nothing.
                    neutral = float(correlation.mean())
                    penalty = np.zeros(self.count)
                    penalty[mask] = texture_weight * (neutral - correlation)
                    distance = distance + penalty
        if simplicity_weight > 0.0:
            distance = distance + simplicity_weight * self.simplicity
        if shape_weight > 0.0:
            covered = self.coverage.get(slot)
            if covered is not None:
                distance = distance + shape_weight * (1.0 - covered)
        prior = getattr(self, "prior", None)
        if prior is not None:
            distance = distance + prior
        best = float(distance.min())
        return np.minimum(distance - best, outlier_ceiling), best


def _observation_costs(
    rgb: np.ndarray,
    descriptor: tuple[float, ...],
    slot: str,
    palette: tuple[PaletteColor, ...],
    face_colors: dict[BlockState, dict[str, np.ndarray]],
    calibration: AppearanceCalibration | None,
    family_priors: dict[str, float] | None,
    slot_matching: bool,
    outlier_ceiling: float,
    light: float = 1.0,
    pattern: np.ndarray | None = None,
    face_textures: dict[str, tuple[list[BlockState], np.ndarray]] | None = None,
    texture_weight: float = 0.0,
) -> tuple[dict[BlockState, float], float]:
    """Score one observation against every palette entry.

    Costs are returned relative to the observation's own best match and capped.
    A sample that landed on the wrong block - because a camera pose is slightly
    off, or something outside the reconstruction occludes it - fits nothing well,
    so every candidate hits the ceiling and the observation abstains instead of
    voting confidently for the wrong answer. The absolute best cost comes back
    too, because it says how much this observation should be trusted at all.
    """
    # Undo the lighting rather than dim every candidate to match it. Dividing the
    # observation back up to how the block would look at full brightness keeps
    # every face on the same footing as the jar's own textures: a face in shadow
    # then has the same room to be wrong as a lit one, where dimming the
    # expectation instead squeezes all the candidates together and makes a dark
    # face weak evidence for anything.
    if light < 1.0:
        rgb = rgb / max(light, MINIMUM_LIGHT_FACTOR)
    texture_cost: dict[BlockState, float] = {}
    if pattern is not None and face_textures and texture_weight > 0.0:
        entry = face_textures.get(slot)
        if entry is not None:
            states, matrix = entry
            correlation = matrix @ pattern
            texture_cost = {
                state: texture_weight * (1.0 - float(value))
                for state, value in zip(states, correlation)
            }
    distances: dict[BlockState, float] = {}
    for item in palette:
        faces = face_colors[item.state]
        candidates = (faces[slot],) if slot_matching and slot in faces else tuple(faces.values())
        if not candidates:
            continue
        distance = min(
            _appearance_distance(rgb, calibration.apply(expected) if calibration else expected)
            for expected in candidates
        )
        if item.descriptors:
            distance += 80.0 * min(
                float(np.linalg.norm(np.asarray(descriptor) - np.asarray(candidate)))
                for candidate in item.descriptors
            )
        if family_priors:
            distance /= max(0.1, family_priors.get(item.family, 1.0))
        # A flat candidate has no pattern to match, so it neither gains nor loses.
        distance += texture_cost.get(item.state, texture_weight)
        distances[item.state] = distance
    if not distances:
        return {}, float("inf")
    best = min(distances.values())
    return {state: min(distance - best, outlier_ceiling) for state, distance in distances.items()}, best


def _support_prior(
    observations: list[tuple[str, BlockCoordinate, np.ndarray, float]],
    index: PaletteIndex,
    weight: float,
) -> np.ndarray:
    """Cost per candidate from how much of the scene it currently explains."""
    wins = np.zeros(index.count)
    for _frame_id, _coordinate, costs, sample_weight in observations:
        if costs.size:
            wins[int(np.argmin(costs))] += max(sample_weight, 0.0)
    total = wins.sum()
    if total <= 0.0:
        return np.zeros(index.count)
    share = np.maximum(wins / total, 1e-4)
    return weight * (-np.log(share) + float(np.log(share.max())))


def _locally_supported(
    observations: list[tuple[str, BlockCoordinate, np.ndarray, float]],
    index: PaletteIndex,
    *,
    region: int = LOCAL_SUPPORT_REGION,
    keep: int = LOCAL_SUPPORT_KEEP,
    minimum_wins: float = LOCAL_SUPPORT_WINS,
) -> set[int]:
    """Candidates that carry a corner of the scene, even if they carry none of it.

    Support pruning ranks candidates over the whole build, so a material used for
    seven blocks of ceiling cannot outrank one used for four hundred, and is
    dropped before it can win anything - which is why a plainly visible row of
    cobblestone came back as a block absent from the scene entirely. Counting wins
    inside coarse regions instead lets a material that dominates one corner survive
    alongside the ones that dominate the whole.
    """
    if not observations or region <= 0 or keep <= 0:
        return set()
    per_region: dict[tuple[int, int, int], dict[int, float]] = {}
    for _frame_id, coordinate, costs, weight in observations:
        if not costs.size or weight <= 0.0:
            continue
        cell = (coordinate[0] // region, coordinate[1] // region, coordinate[2] // region)
        per_region.setdefault(cell, {})
        best = int(np.argmin(costs))
        tally = per_region[cell]
        tally[best] = tally.get(best, 0.0) + weight
    survivors: set[int] = set()
    for tally in per_region.values():
        ranked = sorted(tally, key=lambda position: -tally[position])
        for position in ranked[:keep]:
            if tally[position] >= minimum_wins:
                survivors.add(position)
    return survivors


def _supported(
    observations: list[tuple[str, BlockCoordinate, np.ndarray, float]],
    index: PaletteIndex,
    share: float,
    minimum: int = PALETTE_MINIMUM,
    margin_support: bool = MARGIN_SUPPORT,
    fallback_weight: float = FALLBACK_WEIGHT,
) -> np.ndarray:
    """Indices of candidates winning at least ``share`` of the observed faces.

    The palette offers every full cube in the game, but a build is made of a
    handful of materials. The rest win nothing except the stray faces they happen
    to sit closest to, and every one of those is a block wrong. A floor on how
    many survive keeps a scene from collapsing onto its single most common
    material.
    """
    if not observations:
        return np.arange(index.count)
    wins = np.zeros(index.count)
    for _frame_id, _coordinate, costs, weight in observations:
        if not costs.size:
            continue
        best = int(np.argmin(costs))
        if margin_support and costs.size > 1:
            # Credit a candidate for how decisively it won, not merely for
            # winning. A material the scene really uses beats its rivals on its
            # own faces by a wide margin; a generic candidate that resembles
            # everything wins a great many faces by a hair. Counting wins alone
            # ranks the second above the first, which is how a block absent from
            # the scene survives pruning while a material the scene does use is
            # dropped and has nothing left to win its blocks.
            lowest = np.partition(costs, 1)[:2]
            wins[best] += max(weight, 0.0) * float(lowest.max() - lowest.min())
        else:
            wins[best] += max(weight, 0.0)
    total = wins.sum()
    if total <= 0.0:
        return np.arange(index.count)

    # Pruning by state rather than by material, measured. Grouping a log's three
    # axis states into one material and keeping them together scores 83.68%
    # against 85.35% for keeping the six best states outright: near-duplicate
    # orientations compete for the same faces, so admitting all of a material's
    # states costs more than the material is worth.
    keep = np.flatnonzero(wins >= share * total)
    if keep.size < minimum:
        keep = np.argsort(wins)[::-1][:minimum]
    return keep


def _decide(
    observations: list[tuple[str, BlockCoordinate, np.ndarray, float]],
    index: PaletteIndex,
    adjacency_strength: float,
    adjacency_min_neighbors: int,
    bounded_adjacency: bool,
    evidence_power: float = 0.0,
    coupling_boost: dict[BlockCoordinate, float] | None = None,
    adopt_boost: float = 0.0,
) -> dict[BlockCoordinate, BlockState]:
    """Fuse observations into one block per coordinate, as the final pass does."""
    totals, weights = _accumulate_costs(observations)
    merged: dict[BlockCoordinate, tuple[BlockState, float, set[str], list[BlockState], dict[BlockState, float]]] = {}
    for coordinate, total in totals.items():
        scores = _top_candidates(total / max(weights[coordinate], 1e-6), index)
        if scores:
            best = max(scores, key=scores.get)
            merged[coordinate] = (best, 0.0, set(), [best], scores)
    if adjacency_strength:
        _smooth_adjacent_states(merged, adjacency_strength, adjacency_min_neighbors,
                                bounded=bounded_adjacency, evidence_power=evidence_power,
                                coupling_boost=coupling_boost, adopt_boost=adopt_boost)
    return {
        coordinate: max(values[4], key=values[4].get)
        for coordinate, values in merged.items()
        if values[4]
    }


def _accumulate_costs(
    observations: list[tuple[str, BlockCoordinate, np.ndarray, float]],
    view_weights: dict[str, float] | None = None,
) -> tuple[dict[BlockCoordinate, np.ndarray], dict[BlockCoordinate, float]]:
    """Weighted sum of every observation's cost vector, per coordinate."""
    totals: dict[BlockCoordinate, np.ndarray] = {}
    weights: dict[BlockCoordinate, float] = {}
    for frame_id, coordinate, costs, weight in observations:
        if view_weights is not None:
            weight = weight * view_weights.get(frame_id, 1.0)
        if weight <= 0.0:
            continue
        running = totals.get(coordinate)
        if running is None:
            totals[coordinate] = costs * weight
        else:
            running += costs * weight
        weights[coordinate] = weights.get(coordinate, 0.0) + weight
    return totals, weights


def _top_candidates(
    mean_costs: np.ndarray, index: PaletteIndex, keep: int = 8
) -> dict[BlockState, float]:
    """The cheapest few candidates, negated so callers can keep taking a maximum."""
    if mean_costs.size == 0:
        return {}
    keep = min(keep, mean_costs.size)
    order = np.argpartition(mean_costs, keep - 1)[:keep]
    return {index.states[int(i)]: -float(mean_costs[i]) for i in order}


def _measure_response(
    raw: list[tuple[str, BlockCoordinate, np.ndarray, tuple[float, ...], str, float, np.ndarray | None]],
    decided: dict[BlockCoordinate, BlockState],
    face_colors: dict[BlockState, dict[str, np.ndarray]],
    field: LightField,
) -> LightResponse:
    """Fit how a simulated light level shows up as brightness in these frames.

    The client's brightness setting sits between the game's light levels and the
    pixels, so the mapping is measured from the faces themselves rather than
    assumed: each one contributes the level it should be lit at and how bright it
    actually came out.
    """
    measurements: list[tuple[int, float]] = []
    for _frame_id, coordinate, rgb, _descriptor, slot, _weight, _pattern, _area in raw:
        state = decided.get(coordinate)
        if state is None:
            continue
        expected = face_colors.get(state, {}).get(slot)
        if expected is None:
            continue
        reference = float(np.asarray(expected).mean())
        if reference < 8.0:
            continue
        measurements.append((field.face_level(coordinate, slot), float(rgb.mean()) / reference))
    return fit_response(measurements)


def _view_weights(
    observations: list[tuple[str, BlockCoordinate, np.ndarray, float]],
    index: PaletteIndex,
    strength: float,
) -> dict[str, float]:
    """Trust each frame in proportion to how far it agrees with the others.

    A frame whose camera pose is a little wrong samples across block boundaries,
    so its answers disagree with every other view of the same block. Which frame
    is the odd one out can be settled without ground truth by asking, for blocks
    that several frames can see, how often each frame's own best guess matches
    what the rest of them say.
    """
    frames = {frame_id for frame_id, _c, _costs, _w in observations}
    if len(frames) < 3:
        return {frame_id: 1.0 for frame_id in frames}
    choice: dict[tuple[str, BlockCoordinate], BlockState] = {}
    for frame_id, coordinate, costs, _weight in observations:
        if costs.size == 0:
            continue
        best = index.states[int(np.argmin(costs))]
        key = (frame_id, coordinate)
        if key not in choice:
            choice[key] = best
    by_block: dict[BlockCoordinate, dict[str, BlockState]] = {}
    for (frame_id, coordinate), state in choice.items():
        by_block.setdefault(coordinate, {})[frame_id] = state
    agree: dict[str, list[float]] = {frame_id: [] for frame_id in frames}
    for votes in by_block.values():
        if len(votes) < 2:
            continue
        for frame_id, state in votes.items():
            others = Counter(other for name, other in votes.items() if name != frame_id)
            leader, count = others.most_common(1)[0]
            agree[frame_id].append(1.0 if state == leader and count > 0 else 0.0)
    scores = {
        frame_id: float(np.mean(values)) if values else 1.0 for frame_id, values in agree.items()
    }
    best = max(scores.values()) or 1.0
    # A frame that never agrees still contributes, just faintly.
    return {
        frame_id: float(max(0.05, (value / best) ** strength))
        for frame_id, value in scores.items()
    }


def _palette_face_colors(
    palette: tuple[PaletteColor, ...],
) -> dict[BlockState, dict[str, np.ndarray]]:
    """Expected screen colour of each world face slot, per palette entry."""
    colors: dict[BlockState, dict[str, np.ndarray]] = {}
    for item in palette:
        mapping = {slot: np.asarray(color, dtype=np.float32) for slot, color in item.face_map}
        if not mapping:
            fallback = np.asarray(item.face_rgbs[0] if item.face_rgbs else item.rgb, dtype=np.float32)
            mapping = {slot: fallback for slot in FACE_SLOTS}
        default = next(iter(mapping.values()))
        for slot in FACE_SLOTS:
            mapping.setdefault(slot, default)
        # Vanilla darkens a face by its direction before any block light applies,
        # so an unshaded texture colour never matches what the screenshot shows.
        colors[item.state] = {
            slot: color * FACE_SHADE.get(slot, 1.0) for slot, color in mapping.items()
        }
    return colors


DIRECTIONS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def _slot_brightness(
    raw: list,
    decided: dict[BlockCoordinate, BlockState],
    face_colors: dict[BlockState, dict[str, np.ndarray]],
    index: PaletteIndex,
    texture_weight: float,
    outlier_ceiling: float,
    simplicity_weight: float,
    minimum: int = 12,
) -> dict[str, float]:
    """How bright each facing came out, measured where the classifier chose freely.

    Vanilla's fixed shading is already applied to the expected colours, and yet a
    lit scene still shows a directional bias on top of it - light arrives mostly
    from above, so upward faces run brighter than downward ones by more than the
    constants predict. A light field cannot express that: it gives a cell one
    level, and every face of that cell then shares it.
    """
    ratios: dict[str, list[float]] = {}
    for _frame_id, coordinate, rgb, _descriptor, slot, _weight, pattern, area in raw:
        if area <= 0.0:
            continue
        label = decided.get(coordinate)
        if label is None:
            continue
        expected = face_colors.get(label, {}).get(slot)
        if expected is None:
            continue
        costs, _best = index.costs(
            rgb, (), slot, pattern=pattern, texture_weight=texture_weight,
            outlier_ceiling=outlier_ceiling, simplicity_weight=simplicity_weight,
        )
        if costs.size == 0 or index.states[int(np.argmin(costs))] != label:
            continue
        total = float(np.mean(expected))
        if total > 1e-6:
            ratios.setdefault(slot, []).append(float(np.mean(rgb)) / total)
    return {
        slot: float(np.median(values))
        for slot, values in ratios.items()
        if len(values) >= minimum
    }


def _chroma_key(rgb: np.ndarray, bins: int = CHROMA_BINS) -> tuple[int, int, int]:
    """Which colour a face is, with no regard for how brightly it is lit.

    Dividing by the total discards intensity and keeps the direction, so one
    material lands in one bucket whether it sits under a torch or across the room.
    """
    total = float(np.sum(rgb))
    if total <= 1e-6:
        return (0, 0, 0)
    scaled = np.floor(np.asarray(rgb, dtype=float) / total * bins)
    return tuple(int(v) for v in scaled)  # type: ignore[return-value]


def _scene_spread(raw: list, bins: int = CHROMA_BINS, minimum: int = CAST_GROUP_MINIMUM) -> float:
    """How much one material's brightness varies across the scene.

    A fullbright room paints every face of a material the same colour, up to the
    fixed face shading; a lit room does not, because the same material reads bright
    by a torch and dark across the room. So the spread within a material says
    whether there is any light to undo - and unlike `_scene_brightness` it never
    consults the palette, which is what made that measure read Template 5 as
    fullbright when every material in it has a convincing dark impostor.

    Measured fullbright at 1.35 and the three gamma-100 fixtures at 2.37, 2.45 and
    3.17, so anything near the default threshold separates them.
    """
    groups: dict[tuple[int, int, int], list[float]] = {}
    for _frame_id, _coordinate, rgb, _descriptor, slot, _weight, _pattern, area in raw:
        if area <= 0.0:
            continue
        value = float(np.mean(rgb)) / FACE_SHADE.get(slot, 0.8)
        if value > 1e-6:
            groups.setdefault(_chroma_key(rgb, bins), []).append(value)
    spreads = []
    for values in groups.values():
        if len(values) < minimum:
            continue
        low = float(np.percentile(values, 10.0))
        if low > 1e-6:
            spreads.append(float(np.percentile(values, 90.0)) / low)
    return float(np.median(spreads)) if spreads else 1.0


def _uncast(
    raw: list,
    cast: tuple[float, float, float] = LIGHT_CAST,
    bins: int = CHROMA_BINS,
    minimum: int = CAST_GROUP_MINIMUM,
    floor: float = CAST_FLOOR,
    damping: float = 1.0,
    percentile: float = CAST_PERCENTILE,
    region: int = CAST_REGION,
) -> list:
    """Undo the light on every face, colour as well as brightness.

    Minecraft block light is warm, so a dim scene is not merely darker than a
    bright one - it is redder, and by a fixed amount. Measured against the
    reference, all three gamma-100 fixtures share the same normalised per-channel
    gain to within a few percent while the fullbright one is flat, which makes the
    cast a constant and its magnitude the only unknown. That matters: scaling a
    face by a single number preserves the cast, so a dimmed white block stays
    orange however far it is scaled, and correcting brightness alone was measured
    at under a point of per-face accuracy.

    The magnitude comes from the brightest face of the same colour anywhere in the
    scene. If any face of a material is fully lit, that face is its unlit
    appearance and every dimmer one's shortfall against it is the light it is
    missing - recovered from the pixels, with no palette involved. Measured at
    0.410 against a true 0.414 on Template 6 and 0.331 against 0.316 on Template 4.
    """
    vector = np.asarray(cast, dtype=float)

    # The ceiling is global on purpose. Estimating it per region of the scene is
    # the obvious idea - one end of Template 5 is lit and the other is not - and
    # measured *worse*, taking its diamond crop from 24.33% back to 0.00% at both
    # 16 and 8 blocks. Fewer faces per group means a lower maximum and a
    # correction that under-shoots.
    def bucket(coordinate) -> tuple:
        if region <= 0:
            return ()
        return tuple(int(axis) // region for axis in coordinate)

    gathered: dict[tuple, list[float]] = {}
    for _frame_id, coordinate, rgb, _descriptor, slot, _weight, _pattern, area in raw:
        if area <= 0.0:
            continue
        colour = _chroma_key(rgb, bins)
        value = float(np.mean(rgb)) / FACE_SHADE.get(slot, 0.8)
        gathered.setdefault((colour, bucket(coordinate)), []).append(value)
        if region > 0:
            gathered.setdefault((colour, ()), []).append(value)
    # A maximum is not scale-invariant: over a crop it is taken across four
    # thousand faces and over the whole of Template 5 across ninety thousand, so
    # the larger scene's is systematically higher and over-corrects. A percentile
    # is stable against that; at 100 it is the maximum again.
    groups = {
        colour: (max(values) if percentile >= 100.0
                 else float(np.percentile(values, percentile)))
        for colour, values in gathered.items()
    }
    counts = {colour: len(values) for colour, values in gathered.items()}

    corrected = []
    for observation in raw:
        frame_id, coordinate, rgb, descriptor, slot, weight, pattern, area = observation
        colour = _chroma_key(rgb, bins)
        nearby = (colour, bucket(coordinate))
        key = nearby if counts.get(nearby, 0) >= minimum else (colour, ())
        ceiling = groups.get(key, 0.0)
        magnitude = 1.0
        if counts.get(key, 0) >= minimum and ceiling > 1e-6:
            value = float(np.mean(rgb)) / FACE_SHADE.get(slot, 0.8)
            magnitude = min(1.0, max(floor, value / ceiling))
        # The ceiling estimate over-corrects - 0.262 recovered against a true 0.312
        # on Template 5 - and over-correcting collapses a scene to one pale
        # material. Damping pulls the magnitude back toward one.
        if damping != 1.0:
            magnitude = 1.0 - damping * (1.0 - magnitude)
        lifted = np.clip(np.asarray(rgb, dtype=float) / (magnitude * vector), 0.0, 255.0)
        corrected.append((
            frame_id, coordinate, lifted.astype(np.float32), descriptor, slot,
            weight, pattern, area,
        ))
    return corrected


def _scene_brightness(
    raw: list,
    decided: dict[BlockCoordinate, BlockState],
    face_colors: dict[BlockState, dict[str, np.ndarray]],
    index: PaletteIndex,
    texture_weight: float,
    outlier_ceiling: float,
    simplicity_weight: float,
    applied: float = 1.0,
) -> float:
    """How bright the scene is against the jar, from the faces the classifier owns.

    Comparing how well the palette explains a face cannot tell a fullbright room
    from a dim one - a darker block always exists, so that test always prefers
    leaving the observation alone. This looks instead at the faces where the
    classifier's *own* choice stands, and asks how bright they came out.

    That is not circular, because the cost is weighted toward hue and texture
    rather than brightness. A dim scene's best match is usually the right material
    seen too dark, and the leftover ratio measures the darkness. Fullbright rooms
    come out near one; the dim fixtures come out near a half.
    """
    ratios: list[float] = []
    for _frame_id, coordinate, rgb, _descriptor, slot, _weight, pattern, area in raw:
        if area <= 0.0:
            continue
        label = decided.get(coordinate)
        if label is None:
            continue
        expected = face_colors.get(label, {}).get(slot)
        if expected is None:
            continue
        costs, _best = index.costs(
            rgb, (), slot, pattern=pattern, texture_weight=texture_weight,
            outlier_ceiling=outlier_ceiling, simplicity_weight=simplicity_weight,
        )
        if costs.size == 0 or index.states[int(np.argmin(costs))] != label:
            continue  # the label was imposed by neighbours, not chosen from this face
        total = float(np.mean(expected))
        if total > 1e-6:
            # against what the correction so far has already lifted
            ratios.append(float(np.mean(rgb)) / max(applied, 1e-6) / total)
    return float(np.median(ratios)) if ratios else 1.0


def _explanatory_cost(
    raw: list,
    index: PaletteIndex,
    field: "LightField | None",
    calibrations: dict[str, AppearanceCalibration | None],
    texture_weight: float,
    outlier_ceiling: float,
    simplicity_weight: float,
    sample_size: int = 400,
) -> float:
    """How well the palette explains the faces, under a given light field.

    Whether a scene needs un-lighting is not something to assume. A fullbright room
    already matches the jar, and dimming it against a light field invented from a
    few false sources makes every match worse; a dim room is the reverse. Comparing
    the cost with and without the field says which case this is, using nothing but
    the images.
    """
    step = max(1, len(raw) // max(sample_size, 1))
    total, counted = 0.0, 0
    for observation in raw[::step]:
        frame_id, coordinate, rgb, descriptor, slot, _weight, pattern, area = observation
        if area <= 0.0:
            continue
        factor = field.face_factor(coordinate, slot) if field is not None else 1.0
        _costs, absolute = index.costs(
            rgb, descriptor, slot, calibration=calibrations.get(frame_id), light=factor,
            pattern=pattern, texture_weight=texture_weight,
            outlier_ceiling=outlier_ceiling, simplicity_weight=simplicity_weight,
        )
        total += absolute
        counted += 1
    return total / max(counted, 1)


def _detect_emitters(
    raw: list,
    palette: tuple[PaletteColor, ...],
    face_colors: dict[BlockState, dict[str, np.ndarray]],
    *,
    brightness_floor: float = EMITTER_BRIGHTNESS,
    cost_limit: float = EMITTER_COST,
) -> dict[BlockCoordinate, int]:
    """Find light sources by the one property that survives an unknown exposure.

    Every ordinary block in a dim room is darkened by the light reaching it, and by
    an amount nobody knows - which is why absolute exposure cannot be recovered from
    appearance. A block that makes its own light is not darkened: it renders at its
    own colour whatever the room is doing. On Template 6 the emitting faces sit at
    0.86 of their jar colour against 0.40 for everything else.

    So an emitter is identifiable without knowing the exposure, and finding one
    gives the lighting simulation the sources it otherwise never has - which is the
    whole reason a dim scene classifies as a dark one.

    Only bright emitters are considered. A dim emitter is indistinguishable from an
    ordinary block that happens to be dark, and admitting those costs far more in
    false sources than the few extra true ones are worth.
    """
    glowing = tuple(
        item for item in palette
        if emission_of(item.state) > 0 and float(np.mean(item.rgb)) >= brightness_floor
    )
    if not glowing:
        return {}
    index = PaletteIndex(
        glowing, {item.state: face_colors[item.state] for item in glowing}, None, TEXTURE_WEIGHT
    )
    best: dict[BlockCoordinate, tuple[float, BlockState]] = {}
    for _frame_id, coordinate, rgb, _descriptor, slot, _weight, pattern, area in raw:
        if area <= 0.0:
            continue  # a centre patch is not a measurement of one face
        costs, absolute = index.costs(
            rgb, (), slot, pattern=pattern, texture_weight=TEXTURE_WEIGHT,
            outlier_ceiling=OUTLIER_CEILING, simplicity_weight=SIMPLICITY_WEIGHT,
        )
        if absolute > cost_limit or costs.size == 0:
            continue
        found = index.states[int(np.argmin(costs))]
        if absolute < best.get(coordinate, (np.inf, None))[0]:
            best[coordinate] = (absolute, found)
    return {
        coordinate: emission_of(state) for coordinate, (_cost, state) in best.items()
    }


def _reconcile_exposure(
    raw: list,
    *,
    minimum_pairs: int = 8,
    limit: float = 2.0,
) -> dict[str, float]:
    """Per-frame multipliers that put every frame on one brightness scale.

    How bright a scene *really* is cannot be recovered from appearance alone -
    the palette holds a block for every brightness, so any exposure explains the
    picture and the estimate has nothing to push against. How bright one frame is
    *relative to another* is a different question and a well-posed one: wherever
    two cameras saw the same face, the ratio between what they recorded is a
    direct measurement of the difference between them.

    Solving all those ratios at once gives one gain per frame, fixed only up to a
    common factor, which is exactly the freedom that cannot be resolved. The fit
    is anchored so the gains cancel in the mean, leaving the scene's overall
    brightness untouched and correcting only the disagreement between frames.
    """
    per_face: dict[tuple[BlockCoordinate, str], dict[str, list[float]]] = {}
    for frame_id, coordinate, rgb, _descriptor, slot, _weight, _pattern, area in raw:
        if area <= 0.0:
            continue  # centre patches are not a measurement of one face
        brightness = float(np.mean(rgb))
        if brightness < 4.0:
            continue  # too dark for the ratio to mean anything
        per_face.setdefault((coordinate, slot), {}).setdefault(frame_id, []).append(
            float(np.log(brightness))
        )
    frames = sorted({observation[0] for observation in raw})
    position = {frame_id: index for index, frame_id in enumerate(frames)}
    if len(frames) < 2:
        return {frame_id: 1.0 for frame_id in frames}
    rows: list[np.ndarray] = []
    targets: list[float] = []
    for seen in per_face.values():
        if len(seen) < 2:
            continue
        measured = [(frame_id, float(np.median(values))) for frame_id, values in seen.items()]
        for first in range(len(measured)):
            for second in range(first + 1, len(measured)):
                row = np.zeros(len(frames))
                row[position[measured[first][0]]] = 1.0
                row[position[measured[second][0]]] = -1.0
                rows.append(row)
                targets.append(measured[first][1] - measured[second][1])
    if len(rows) < minimum_pairs:
        return {frame_id: 1.0 for frame_id in frames}
    rows.append(np.ones(len(frames)))
    targets.append(0.0)
    solution, *_rest = np.linalg.lstsq(
        np.asarray(rows), np.asarray(targets, dtype=float), rcond=None
    )
    ceiling = float(np.log(max(limit, 1.0 + 1e-6)))
    return {
        frame_id: float(np.exp(-np.clip(solution[index], -ceiling, ceiling)))
        for frame_id, index in position.items()
    }


def _disagreement(
    observations: list[tuple[str, BlockCoordinate, np.ndarray, float]],
    index: PaletteIndex,
) -> dict[BlockCoordinate, float]:
    """Per block, the share of its evidence that dissents from the majority view.

    Whether the cameras that saw a block agree about it turns out to be the
    sharpest confidence signal available: on Template 2 blocks their views agree
    on are 94% right, blocks they split over 72%. Keying this on agreement rather
    than on how many cameras looked matters, because a block only one camera saw
    trivially agrees with itself and keeps its say - counting views instead
    punishes exactly the thin, singly-observed log inlays this project keeps
    losing.
    """
    votes: dict[BlockCoordinate, dict[BlockState, float]] = {}
    for _frame_id, coordinate, costs, weight in observations:
        if costs.size == 0 or weight <= 0.0:
            continue
        winner = index.states[int(np.argmin(costs))]
        tally = votes.setdefault(coordinate, {})
        tally[winner] = tally.get(winner, 0.0) + weight
    shares: dict[BlockCoordinate, float] = {}
    for coordinate, tally in votes.items():
        total = sum(tally.values())
        if total > 0.0:
            shares[coordinate] = 1.0 - max(tally.values()) / total
    return shares


def _coupling_boost(
    observations: list[tuple[str, BlockCoordinate, np.ndarray, float]],
    index: PaletteIndex,
    upward_share: dict[BlockCoordinate, float] | None,
    fallback_share: dict[BlockCoordinate, float] | None,
    upward_risk: float,
    fallback_risk: float,
    disagreement_risk: float,
) -> dict[BlockCoordinate, float]:
    """How much harder each block should listen to its neighbours.

    Three independent reasons a block's own appearance may be describing
    something other than itself - it was seen only from above, where carpets and
    slabs sit; it was never resolved as a face at all; or the cameras that saw it
    disagree. They are summed rather than kept apart because they answer the same
    question and the smoothing only needs one number per block.
    """
    boost: dict[BlockCoordinate, float] = {}
    if upward_risk and upward_share:
        for coordinate, share in upward_share.items():
            boost[coordinate] = boost.get(coordinate, 0.0) + upward_risk * share
    if fallback_risk and fallback_share:
        for coordinate, share in fallback_share.items():
            boost[coordinate] = boost.get(coordinate, 0.0) + fallback_risk * share
    if disagreement_risk:
        for coordinate, share in _disagreement(observations, index).items():
            boost[coordinate] = boost.get(coordinate, 0.0) + disagreement_risk * share
    return boost


def _smooth_adjacent_states(
    merged: dict[BlockCoordinate, tuple[BlockState, float, set[str], list[BlockState], dict[BlockState, float]]],
    strength: float,
    minimum_neighbors: int,
    *,
    bounded: bool = False,
    iterations: int = 6,
    evidence_power: float = 0.0,
    coupling_boost: dict[BlockCoordinate, float] | None = None,
    adopt_boost: float = 0.0,
) -> None:
    """Settle labels with iterated conditional modes over each block's own candidates.

    Colour alone cannot separate a plank face from the bark face of a log, but
    builds are locally uniform, so a block's neighbours are real evidence. Each
    block may only move among the candidates its own observations proposed, and
    the coupling is scaled by how decisive the appearance costs actually were, so
    a confident block is not talked out of its answer by its neighbours.
    """
    if not merged or strength <= 0.0:
        return
    labels = {coordinate: max(values[4], key=values[4].get) for coordinate, values in merged.items()}
    margins = []
    for values in merged.values():
        ranked = sorted(values[4].values(), reverse=True)
        if len(ranked) > 1:
            margins.append(ranked[0] - ranked[1])
    scale = float(np.median(margins)) if margins else 1.0
    coupling = strength * max(scale, 1e-6)
    # Fusing observations into a mean throws away how many there were, so a block
    # one camera glimpsed argues as loudly as one five cameras agreed on. That is
    # exactly backwards where it matters most: a floor tile's upward face is the
    # only face it has, and when a carpet or a slab sits on it - geometry too
    # thin for the reconstruction to hold, so it cannot be culled as an occluder
    # - that single face is a picture of the covering block, confidently wrong.
    # Letting a block's resistance to its neighbours follow the number of frames
    # that actually saw it puts those tiles back under the influence of the floor
    # they belong to, without loosening blocks that were seen properly.
    seen_counts = [len(values[2]) for values in merged.values() if values[2]]
    typical_views = float(np.median(seen_counts)) if seen_counts else 1.0

    def coupling_for(coordinate, values) -> float:
        local = coupling
        if evidence_power > 0.0:
            views = max(len(values[2]), 1)
            local = local * (typical_views / views) ** evidence_power
        if coupling_boost:
            local = local * (1.0 + coupling_boost.get(coordinate, 0.0))
        return local

    for _iteration in range(max(1, iterations)):
        changed = 0
        for coordinate, values in merged.items():
            neighbours = [
                labels[neighbour]
                for neighbour in (
                    (coordinate[0] + dx, coordinate[1] + dy, coordinate[2] + dz)
                    for dx, dy, dz in DIRECTIONS
                )
                if neighbour in labels
            ]
            if len(neighbours) < minimum_neighbors:
                continue
            counts = Counter(neighbours)
            local = coupling_for(coordinate, values)
            options = values[4]
            if adopt_boost > 0.0 and coupling_boost:
                # A cell may only ever re-rank its *own* candidates, so coupling
                # cannot rescue one whose evidence never proposed the right answer
                # - which is exactly the centre-patch fallback cells, and why
                # `fallback_risk` measured completely inert. Where the risk is high
                # enough, let the neighbours' majority stand as a candidate in its
                # own right, scored at the cell's own worst.
                if coupling_boost.get(coordinate, 0.0) >= adopt_boost and counts:
                    majority, _votes = counts.most_common(1)[0]
                    if majority not in options:
                        options = dict(options)
                        options[majority] = min(options.values()) if options else 0.0
            best_state, best_score = labels[coordinate], None
            for candidate, score in options.items():
                total = score + local * counts.get(candidate, 0)
                if best_score is None or total > best_score:
                    best_state, best_score = candidate, total
            if best_state != labels[coordinate]:
                labels[coordinate] = best_state
                changed += 1
        if changed == 0:
            break
    for coordinate, state in labels.items():
        values = merged[coordinate]
        # Make the settled label the winner for the caller's argmax.
        values[4][state] = max(values[4].values()) + 1.0


def _choose_state_with_local_family_support(
    coordinate: BlockCoordinate,
    candidate_scores: dict[BlockState, float],
    merged: dict[BlockCoordinate, tuple[BlockState, float, set[str], list[BlockState], dict[BlockState, float]]],
    palette_families: dict[BlockState, str],
) -> BlockState | None:
    if not candidate_scores:
        return None
    return max(candidate_scores, key=candidate_scores.get)


def _state_variant_prefix(state: BlockState) -> str:
    raw = state.name.rsplit(":", 1)[-1]
    if "_" not in raw:
        return raw
    pieces = raw.split("_")
    suffixes = {"log", "wood", "planks", "slab", "stairs", "brick", "stone", "sandstone", "quartz", "terracotta"}
    if pieces[-1] in suffixes:
        return "_".join(pieces[:-1])
    for index in range(len(pieces) - 1, 0, -1):
        if pieces[index] in suffixes:
            return "_".join(pieces[:index])
    return "_".join(pieces[:2])


def _remove_enclosed_blocks(volume: BlockVolume) -> None:
    occupied = set(volume.blocks)
    directions = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    enclosed = {
        coordinate
        for coordinate in occupied
        if all(
            (coordinate[0] + dx, coordinate[1] + dy, coordinate[2] + dz) in occupied
            for dx, dy, dz in directions
        )
    }
    for coordinate in enclosed:
        del volume.blocks[coordinate]


def _block_center(coordinate: tuple[int, int, int]) -> tuple[float, float, float]:
    return tuple(value + 0.5 for value in coordinate)


def _nearest_palette(
    rgb: np.ndarray,
    palette: tuple[PaletteColor, ...],
    appearance_calibration: AppearanceCalibration | None = None,
) -> tuple[BlockState, float, float]:
    distances = [
        min(
            _appearance_distance(
                rgb,
                appearance_calibration.apply(np.asarray(face_rgb, dtype=np.float32))
                if appearance_calibration
                else np.asarray(face_rgb, dtype=np.float32),
            )
            for face_rgb in (tuple(color for _, color in item.face_map) or item.face_rgbs or (item.rgb,))
        )
        for item in palette
    ]
    ordered = sorted(distances)
    index = distances.index(ordered[0])
    margin = ordered[1] - ordered[0] if len(ordered) > 1 else ordered[0]
    return palette[index].state, ordered[0], margin


def _appearance_distance(
    observed: np.ndarray, expected: np.ndarray, *, minimum_light: float | None = None
) -> float:
    """Compare colour allowing for how dark Minecraft's block lighting can make a face.

    A texture is only ever darkened by lighting, never brightened past its own
    colour, so the expected colour may be scaled by a gain in
    ``[minimum_light, 1]`` before the comparison. Without that, bright blocks are
    punished for standing in shadow far more than dark ones: a birch floor in a
    dim corner is further from birch, in plain RGB, than dark oak is. Chroma is
    already gain free and carries the hue.
    """
    if minimum_light is None:
        minimum_light = MINIMUM_LIGHT
    expected_energy = float(expected @ expected)
    if expected_energy > 1e-6 and minimum_light < 1.0:
        gain = float(np.clip(float(observed @ expected) / expected_energy, minimum_light, 1.0))
    else:
        gain = 1.0
    rgb_distance = float(np.linalg.norm(observed - gain * expected))
    observed_total = float(observed.sum())
    expected_total = float(expected.sum())
    if observed_total <= 1e-6 or expected_total <= 1e-6:
        return rgb_distance
    observed_chroma = observed / observed_total
    expected_chroma = expected / expected_total
    chroma_distance = float(np.linalg.norm(observed_chroma - expected_chroma)) * 441.67
    return (1.0 - CHROMA_WEIGHT) * rgb_distance + CHROMA_WEIGHT * chroma_distance


def _candidate_face_distance(
    rgb: np.ndarray,
    item: PaletteColor,
    appearance_calibration: AppearanceCalibration | None = None,
    state_appearance_profiles: dict[BlockState, tuple[tuple[int, int, int], ...]] | None = None,
    observed_texture: np.ndarray | None = None,
) -> float:
    profiles = state_appearance_profiles.get(item.state, ()) if state_appearance_profiles else ()
    colors = profiles or tuple(color for _, color in item.face_map) or item.face_rgbs or (item.rgb,)
    if not colors:
        return float("inf")
    color_distance = min(
        _appearance_distance(
            rgb,
            appearance_calibration.apply(np.asarray(color, dtype=np.float32))
            if appearance_calibration and not profiles
            else np.asarray(color, dtype=np.float32),
        )
        for color in colors
    )
    if observed_texture is None or not item.face_thumbnails:
        return color_distance
    texture_distance = min(
        _texture_pattern_distance(observed_texture, encoded)
        for _, encoded in item.face_thumbnails
    )
    return color_distance + texture_distance * 120.0


def _palette_candidates(
    rgb: np.ndarray,
    palette: tuple[PaletteColor, ...],
    appearance_calibration: AppearanceCalibration | None = None,
    state_appearance_profiles: dict[BlockState, tuple[tuple[int, int, int], ...]] | None = None,
    family_priors: dict[str, float] | None = None,
    descriptor: tuple[float, ...] | None = None,
    *,
    limit: int,
    observed_texture: np.ndarray | None = None,
) -> list[tuple[BlockState, float, float]]:
    distances = []
    for item in palette:
        distance = _candidate_face_distance(
            rgb,
            item,
            appearance_calibration=appearance_calibration,
            state_appearance_profiles=state_appearance_profiles,
            observed_texture=observed_texture,
        )
        if descriptor and item.descriptors:
            descriptor_distance = min(
                float(np.linalg.norm(np.asarray(descriptor) - np.asarray(candidate)))
                for candidate in item.descriptors
            )
            distance += descriptor_distance * 80.0
        if family_priors:
            distance /= max(0.1, family_priors.get(item.family, 1.0))
        distances.append(distance)
    order = sorted(range(len(distances)), key=distances.__getitem__)[:limit]
    return [
        (
            palette[index].state,
            distances[index],
            distances[order[1]] - distances[index] if len(order) > 1 else distances[index],
        )
        for index in order
    ]


def _texture_pattern_distance(observed: np.ndarray, encoded: str) -> float:
    try:
        import base64
        from io import BytesIO
        from PIL import Image

        expected = np.asarray(Image.open(BytesIO(base64.b64decode(encoded))), dtype=np.float32).reshape(-1, 3)
    except (ValueError, OSError, TypeError):
        return float("inf")
    actual = observed.astype(np.float32).reshape(-1, 3)
    if actual.shape != expected.shape:
        return float("inf")
    actual_centered = actual - actual.mean(axis=0)
    expected_centered = expected - expected.mean(axis=0)
    actual_scale = np.maximum(actual_centered.std(axis=0), 1.0)
    expected_scale = np.maximum(expected_centered.std(axis=0), 1.0)
    return float(np.linalg.norm(actual_centered / actual_scale - expected_centered / expected_scale) / np.sqrt(actual.size))


def _patch_descriptor(patch: np.ndarray, coverage: np.ndarray | None = None) -> tuple[float, ...]:
    pixels = patch.astype(np.float32).reshape(-1, 3)
    weights = None if coverage is None else coverage.reshape(-1)
    normalized = pixels / np.maximum(pixels.sum(axis=1, keepdims=True), 1.0)
    histogram = np.histogram(normalized[:, 0], bins=4, range=(0.0, 1.0), weights=weights, density=True)[0]
    luminance = pixels.mean(axis=1)
    average = float(np.average(luminance, weights=weights)) if weights is not None and float(weights.sum()) > 1e-6 else float(luminance.mean())
    variance = (float(np.average((luminance - average) ** 2, weights=weights))
                if weights is not None and float(weights.sum()) > 1e-6 else float(luminance.var()))
    if patch.ndim == 2:
        gradients = np.abs(np.diff(luminance)).mean() if len(luminance) > 1 else 0.0
    else:
        gradients = np.abs(np.diff(patch.astype(np.float32).mean(axis=2), axis=0)).mean()
    return tuple(float(value) for value in (*histogram, average / 255.0, np.sqrt(variance) / 255.0, gradients / 255.0))


def _rectify_projected_face(
    image: np.ndarray,
    projected_corners: np.ndarray,
    output_size: int = 16,
) -> np.ndarray | None:
    """Warp a projected face quadrilateral into a square, face-on patch."""
    if projected_corners.shape != (4, 2) or output_size < 2:
        raise ValueError("projected_corners must have shape (4, 2) and output_size must be at least 2")
    return rectify_face(image, projected_corners, output_size)


def _dominant_face_patch(
    coordinate: BlockCoordinate,
    camera: CameraModel,
    scene: SceneTransform,
    image: np.ndarray,
    front_pixel: tuple[int, int],
) -> tuple[np.ndarray, float] | None:
    center = np.asarray(_block_center(scene.to_world(coordinate)), dtype=np.float64)
    eye = np.asarray(camera.position, dtype=np.float64) + np.array([0.0, camera.eye_height, 0.0])
    direction = eye - center
    axis = int(np.argmax(np.abs(direction)))
    sign = 1.0 if direction[axis] > 0.0 else -1.0
    if axis == 0:
        normal = (sign, 0.0, 0.0)
        offsets = ((sign * 0.5, -0.5, -0.5), (sign * 0.5, -0.5, 0.5), (sign * 0.5, 0.5, 0.5), (sign * 0.5, 0.5, -0.5))
    elif axis == 1:
        normal = (0.0, sign, 0.0)
        offsets = ((-0.5, sign * 0.5, -0.5), (0.5, sign * 0.5, -0.5), (0.5, sign * 0.5, 0.5), (-0.5, sign * 0.5, 0.5))
    else:
        normal = (0.0, 0.0, sign)
        offsets = ((-0.5, -0.5, sign * 0.5), (0.5, -0.5, sign * 0.5), (0.5, 0.5, sign * 0.5), (-0.5, 0.5, sign * 0.5))
    corners = project_points([center + offset for offset in offsets], camera, width=image.shape[1], height=image.shape[0])
    if any(corner is None for corner in corners):
        return None
    projected = np.asarray([(corner[0], corner[1]) for corner in corners if corner is not None])
    center_projection = project_points([center], camera, width=image.shape[1], height=image.shape[0])[0]
    if center_projection is None or (round(center_projection[0]), round(center_projection[1])) != front_pixel:
        return None
    edge_a = projected[1] - projected[0]
    edge_b = projected[3] - projected[0]
    area = abs(float(edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0]))
    if area < 6.0:
        return None
    rectified = _rectify_projected_face(image, projected, 16)
    if rectified is not None:
        return rectified.astype(np.float32).reshape(-1, 3), area
    samples = []
    for v in (0.25, 0.5, 0.75):
        for u in (0.25, 0.5, 0.75):
            point = ((1 - u) * (1 - v) * projected[0] + u * (1 - v) * projected[1] + u * v * projected[2] + (1 - u) * v * projected[3])
            sample_x, sample_y = round(float(point[0])), round(float(point[1]))
            if 0 <= sample_x < image.shape[1] and 0 <= sample_y < image.shape[0]:
                samples.append(image[sample_y, sample_x].astype(np.float32))
    return (np.asarray(samples, dtype=np.float32), area) if len(samples) >= 3 else None


