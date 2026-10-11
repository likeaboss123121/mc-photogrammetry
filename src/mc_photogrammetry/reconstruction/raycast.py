"""Block-state recovery by calibrated projection, z-buffering and appearance scoring.

``reconstruct_scene_surfaces`` can optionally write a *stage trace* into the
caller-supplied ``stage_trace`` mapping. The trace records reference-agnostic
facts about the pipeline - which canonical palette states existed, which states
survived support pruning in each round (and, per coordinate, which of that
cell's own ranked candidates did), where each coordinate's appearance evidence
came from, what the top candidates and their raw pre-smoothing winner were, what
adjacency smoothing settled on and what was finally placed. Membership is keyed
by state, never by some current winner, so the evaluator can ask after the fact
whether the reference state was still reachable. It is a pure observation of the
run: nothing in it is read back into the decisions, and the traced run is
byte-identical to the untraced one. Joining the trace to reference labels,
visibility or masks is the evaluator's job, after reconstruction has finished.

Trace ``schema`` 2 keeps every schema-1 key and meaning and adds three
observation-level, output-neutral keys. ``observations`` records each retained
raw observation in order with its sampler scalars and shape/dtype-bound SHA-256
digests of the rectified patch, masks, colour, descriptor and pattern.
``observation_costs`` records the run's final scored pass - never the
intermediate calibration or pruning passes - with the palette order, the
candidate cost vector, the absolute best cost and the applied weights.
``aggregation`` records the exact ``_accumulate_costs`` totals and weights per
coordinate before any candidate filter, so selection can be replayed. Nothing
in them is read back, and no raw pixels are serialized.

A separate, opt-in ``stage_sidecar`` mapping can carry the *actual* sampled
pixels the trace deliberately omits: the final retained ``resolved_face``
observations' raw sampler patch, scorer-facing ``observed_patch``, validity mask
and coverage, as lossless base64/``dtype``/``shape`` records keyed to the trace's
observation ordinals and digests. The raw sampler patch is what E1 stores and the
observed patch is what the trace digests, so carrying both lets an evaluator
establish exact raw-patch parity and still tie the observed digest to E2A. It is
populated only when a caller passes both ``stage_trace`` and ``stage_sidecar``,
is read back by nothing in production, and is refused rather than truncated past
its record or byte bound.
"""
from __future__ import annotations

import base64
import hashlib
from collections import Counter
from collections.abc import Callable, Mapping
from threading import Event

import numpy as np

from mc_photogrammetry.domain.models import BlockCoordinate, BlockObservation, BlockState, BlockVolume
from mc_photogrammetry.blocks.appearance import (
    CHROMA_WEIGHT,
    DEFAULT_PALETTE,
    MINIMUM_LIGHT,
    PaletteColor,
    appearance_distance as _appearance_distance,
    candidate_face_distance as _candidate_face_distance,
    normalised_pattern as _normalised_pattern,
    palette_candidates as _palette_candidates,
    palette_face_colors as _palette_face_colors,
    patch_descriptor as _patch_descriptor,
    slot_coverage,
    texture_pattern_distance as _texture_pattern_distance,
)
from mc_photogrammetry.blocks.model_atlas import FACE_SHADE, FACE_SLOTS
from mc_photogrammetry.vision.gui_mask import create_gui_mask
from .appearance_scoring import (
    MINIMUM_LIGHT_FACTOR,
    PaletteIndex,
    observation_costs as _observation_costs,
    palette_face_textures as _palette_face_textures,
)
from .camera import CameraModel, project_points
from .face_sampling import (
    DEPTH_SCALE,
    MINIMUM_FACE_AREA,
    PATCH_SIZE,
    PASS_SURFACE,
    PASS_SURFACE_PROPOSED,
    SamplingGeometryDecision,
    exact_sampling_blocks,
    rectify_face,
    resolve_sampling_geometry,
    sample_visible_faces,
    validate_sampling_states,
)
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

STAGE_TRACE_CANDIDATE_LIMIT = 8
"""How many of a coordinate's cheapest candidates the optional stage trace keeps.

Diagnostic-only: the trace's per-coordinate candidate list is truncated to this
many entries (best first), and the truncation is recorded in the trace header so
a reader can tell a full list from a clipped one. A value below one is rejected.

The same limit also bounds the trace-only ``ranked_candidates`` list: derived from
the same per-cell mean costs through the same ``_top_candidates`` helper, it is
recorded as deep as the limit reaches while production selection keeps reading
its unchanged top-K map. A benchmark caller may raise it (for example to sweep
candidate recall at K = 16 or 32) because it only widens a diagnostically
derived copy of the cheapest-first order: the winner, the map smoothing and
placement consume, and every later decision stay identical.
"""

STAGE_SIDECAR_SCHEMA = "mc-photogrammetry-texture-patch-sidecar-v1"
"""Schema tag of the opt-in resolved-face payload sidecar.

The sidecar is written beside the E2A stage trace, never inside it. It carries
the production sampler's own raw patch, its scorer-facing ``observed_patch`` and
its ``valid``/``coverage`` masks for every final retained ``resolved_face``
observation, keyed by the trace observation ordinal and using the same
shape/dtype-bound digest contract. The raw patch is the array E1 retains; the
observed patch is the array the E2A trace digests.
"""

STAGE_SIDECAR_MAX_RECORDS = 10_000
"""Hard cap on resolved-face payload records; above it the sidecar is refused.

This is a refusal bound, not a truncation: a run whose sampler retains more
resolved faces than this gets no sidecar at all, so a reader can never mistake a
clipped payload for the complete sampled set.
"""

STAGE_SIDECAR_MAX_BYTES = 64 * 1024 * 1024
"""Hard cap, in bytes, on the serialized sidecar; above it the sidecar is refused."""

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


def _pixel_authority(
    frames: Mapping[str, np.ndarray],
    usable_masks: Mapping[str, np.ndarray] | None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Normalise the usable-pixel domain once for one reconstruction call.

    Returns ``(usable, excluded)`` keyed by frame, both in source-image
    coordinates. Every consumer in `reconstruct_scene_surfaces` - face sampling,
    residual proposals and partial fitting - reads this one authority, so the
    thin-block path can no longer sample under the GUI mask and then score
    residuals over unmasked pixels.

    ``usable_masks is None`` derives the authority from ``~create_gui_mask`` per
    frame, which preserves the sampling domain the reconstruction already used.
    A supplied mapping is authoritative instead: a frame it omits is unusable
    (all false), and a mask whose shape does not match its frame raises before
    anything is sampled or scored.
    """
    usable: dict[str, np.ndarray] = {}
    excluded: dict[str, np.ndarray] = {}
    for frame_id, image in frames.items():
        height, width = image.shape[:2]
        if usable_masks is None:
            mask = ~create_gui_mask(height, width)
        else:
            supplied = usable_masks.get(frame_id)
            if supplied is None:
                mask = np.zeros((height, width), dtype=bool)
            else:
                mask = np.asarray(supplied, dtype=bool)
                if mask.shape != (height, width):
                    raise ValueError("usable mask must match the image dimensions")
        usable[frame_id] = mask
        excluded[frame_id] = ~mask
    return usable, excluded


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
    sampling_states: Mapping[BlockCoordinate, BlockState] | None = None,
    stage_trace: dict[str, object] | None = None,
    stage_trace_candidate_limit: int = STAGE_TRACE_CANDIDATE_LIMIT,
    stage_sidecar: dict[str, object] | None = None,
) -> BlockVolume:
    """Recover visible scene-grid blocks by calibrated projection and z-buffering.

    ``stage_trace``, when supplied, is filled with a JSON-safe, evaluator-only
    record of the pipeline's own intermediate decisions. It never changes what
    the pipeline decides: no traced value is read back into sampling, scoring,
    pruning, smoothing or placement. ``stage_trace_candidate_limit`` only widens
    the evaluator's copy of that cheapest-first order; the winner and the map
    every later stage consumes are the unchanged production top-K map.

    ``stage_sidecar``, when supplied, must accompany a ``stage_trace`` and is
    filled with the actual resolved-face sampled payloads (the raw sampler patch,
    the scorer-facing observed patch, the validity mask and the coverage) keyed to
    the trace's observation ordinals. It is an evaluator-only artifact: nothing in
    this module reads it back, and passing it changes no sampling, scoring,
    pruning, smoothing, placement or diagnostic.
    """
    if sample_radius < 0:
        raise ValueError("sample_radius cannot be negative")
    if stage_trace_candidate_limit < 1:
        raise ValueError("stage_trace_candidate_limit must be positive")
    if stage_sidecar is not None and stage_trace is None:
        raise ValueError(
            "stage_sidecar requires stage_trace; the sidecar is keyed to the "
            "schema-2 trace's observation ordinals and digests"
        )
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
    # An existing-hypothesis snapshot is validated once, up front. A bad mapping
    # type, coordinate or value raises here, before anything is sampled or scored,
    # and nothing below invents a state, admits a coordinate the reconstruction
    # did not already see, or lets the snapshot reach the material scores.
    sampling_states = validate_sampling_states(sampling_states)
    # One usable-pixel authority for the whole call, normalised here so a
    # wrong-shaped mask is rejected before any frame is sampled or scored.
    usable_pixels, excluded_pixels = _pixel_authority(frames, usable_masks)
    palette_families = {item.state: item.family for item in palette}
    face_colors = _palette_face_colors(palette)
    trace_enabled = stage_trace is not None
    # Evaluation-only side tables. They are populated only when a trace was
    # requested and are never read back into sampling, scoring, pruning or
    # placement; ``trace_observations`` stays index-aligned with ``raw``.
    trace_observations: list[dict[str, object]] = []
    trace_best: list[float] = []
    # Sidecar-only payload records. ``None`` - not an empty list - keeps the
    # ordinary traced and untraced paths free of any payload container: with no
    # opt-in there is nothing to allocate, copy, digest or encode, and the
    # ``_sidecar_record`` helper is unreachable.
    sidecar_records: list[dict[str, object]] | None = (
        [] if stage_sidecar is not None else None
    )
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
            gui_masks=excluded_pixels,
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
    #
    # Geometry is resolved once per call, not once per frame: the target set does
    # not depend on the frame, and one call must be entirely exact-model or
    # entirely cube so its single depth buffer can never mix the two. The snapshot
    # is consulted only for the targets this call resolves; entries outside them
    # are ignored and never become sampling targets themselves.
    surface_decision = resolve_sampling_geometry(
        surface, sampling_states, palette, PASS_SURFACE)
    surface_blocks = exact_sampling_blocks(surface_decision, surface, sampling_states)
    proposed_decision: SamplingGeometryDecision | None = None
    proposed_blocks: dict[BlockCoordinate, BlockState] | None = None
    if proposed:
        proposed_targets = proposed | surface
        proposed_decision = resolve_sampling_geometry(
            proposed_targets, sampling_states, palette, PASS_SURFACE_PROPOSED)
        proposed_blocks = exact_sampling_blocks(proposed_decision, proposed_targets, sampling_states)
    if diagnostics is not None:
        # Stored as JSON-safe records, not the live frozen decisions: diagnostics
        # are serialized (and diffed) by callers, and the typed objects are not.
        # Both executed passes are recorded in order, one entry when no proposal
        # pass ran; a pass that did not run gets no synthetic record.
        executed = (
            (surface_decision,)
            if proposed_decision is None
            else (surface_decision, proposed_decision)
        )
        diagnostics["sampling_geometry"] = tuple(
            decision.to_record() for decision in executed
        )
    # Only an exact decision carries blocks/palette into the sampler. A cube
    # decision passes neither, so the legacy ownership buffer is untouched.
    surface_geometry = {} if surface_blocks is None else {"blocks": surface_blocks, "palette": palette}
    proposed_geometry = (
        {} if proposed_blocks is None else {"blocks": proposed_blocks, "palette": palette}
    )
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
        gui_mask = excluded_pixels[frame_id]
        found = list(sample_visible_faces(
            surface, camera, scene, image, gui_mask=gui_mask,
            minimum_area=minimum_face_area, shrink=face_shrink,
            minimum_visible_fraction=minimum_visible_fraction,
            rejections=face_rejections, depth_scale=depth_scale,
            **surface_geometry,
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
                **proposed_geometry,
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
            rgb = sample.rgb
            descriptor = _patch_descriptor(observed_patch, sample.coverage)
            pattern = _normalised_pattern(observed_patch, sample.coverage)
            raw.append((
                frame_id, sample.coordinate, rgb, descriptor, sample.slot, weight,
                pattern, sample.area,
            ))
            if trace_enabled:
                trace_observations.append(_observation_record(
                    ordinal=len(raw) - 1,
                    frame_id=frame_id, coordinate=sample.coordinate,
                    slot=sample.slot, source="resolved_face",
                    area=float(sample.area), depth=float(sample.depth),
                    visible=float(sample.visible), facing=float(sample.facing),
                    weight=weight, observed_patch=observed_patch,
                    valid=sample.valid, coverage=sample.coverage,
                    rgb=rgb, descriptor=descriptor, pattern=pattern,
                ))
                if sidecar_records is not None:
                    # The opt-in payload capture: the sampler's own arrays, read
                    # once here (never mutated) into lossless array records. Both
                    # the raw ``sample.patch`` and its scorer-facing
                    # ``observed_patch`` are carried, so an evaluator can prove
                    # exact E1 raw-patch parity and still tie the observed digest
                    # to E2A. A centre-patch fallback has no rectified face and is
                    # refused this branch entirely, so the sidecar carries
                    # resolved faces only and its ordinals stay a subset of the
                    # trace's.
                    #
                    # The fixed record bound is enforced *here*, before the extra
                    # record's arrays are copied, digested or base64-encoded, so
                    # an over-limit run is refused outright instead of
                    # accumulating a payload the serializer would only reject
                    # later. The serializer keeps its own bound check as a second
                    # line of defence for direct callers.
                    if len(sidecar_records) >= STAGE_SIDECAR_MAX_RECORDS:
                        raise ValueError(
                            "stage sidecar exceeds the "
                            f"{STAGE_SIDECAR_MAX_RECORDS} resolved-face record "
                            "bound; refusing rather than truncating"
                        )
                    sidecar_records.append(_sidecar_record(
                        ordinal=len(raw) - 1,
                        frame_id=frame_id, coordinate=sample.coordinate,
                        slot=sample.slot, area=float(sample.area),
                        depth=float(sample.depth), visible=float(sample.visible),
                        facing=float(sample.facing), weight=weight,
                        patch=sample.patch, observed_patch=observed_patch,
                        valid=sample.valid, coverage=sample.coverage,
                    ))

    # Pass three: blocks whose faces were all too small, clipped or hidden behind
    # the HUD still deserve an answer, so fall back to their centre patch.
    covered = {coordinate for _f, coordinate, _c, _d, _s, _w, _p, _a in raw}
    for frame_id, image in frames.items():
        frame_usable = usable_pixels[frame_id]
        for coordinate, (depth, pixel) in per_frame_front[frame_id].items():
            if coordinate in covered:
                continue
            x, y = pixel
            # The whole centre patch is the evidence, so one unusable source
            # pixel disqualifies the fallback. Otherwise excluded RGB would
            # still reach the palette classifier, the descriptor and the volume
            # through this pass - the authority would be a lie for exactly the
            # blocks no rectified face could vouch for.
            if not frame_usable[
                y - sample_radius : y + sample_radius + 1,
                x - sample_radius : x + sample_radius + 1,
            ].all():
                continue
            patch = image[y - sample_radius : y + sample_radius + 1, x - sample_radius : x + sample_radius + 1]
            pixels = patch.astype(np.float32).reshape(-1, 3)
            if pixels.size == 0 or float(pixels.std()) < min_surface_contrast:
                continue
            # The centre patch is not a rectified face, so it carries no pattern.
            rgb = np.median(pixels, axis=0)
            descriptor = _patch_descriptor(patch)
            slot = _dominant_face_slot(coordinate, cameras[frame_id], scene)
            weight = fallback_weight / (1.0 + depth / 16.0)
            raw.append((frame_id, coordinate, rgb, descriptor, slot, weight, None, 0.0))
            if trace_enabled:
                trace_observations.append(_observation_record(
                    ordinal=len(raw) - 1,
                    frame_id=frame_id, coordinate=coordinate, slot=slot,
                    source="centre_patch_fallback",
                    area=0.0, depth=float(depth), visible=None, facing=None,
                    weight=weight, observed_patch=None, valid=None, coverage=None,
                    rgb=rgb, descriptor=descriptor, pattern=None,
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

    if trace_enabled:
        # Reference-agnostic provenance for every coordinate the pipeline has
        # any appearance evidence about. Recorded before the light/cast and
        # exposure passes below so it describes the sampled input itself.
        stage_trace["coordinate_evidence"] = _evidence_records(raw)

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
        if trace_enabled:
            # Only the run's final scored pass is kept; each call starts over.
            trace_best.clear()
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
            if trace_enabled:
                trace_best.append(float(best))
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
    prune_rounds: list[dict[str, object]] = []
    for round_index in range(palette_prune_rounds if palette_support > 0.0 else 0):
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
        if trace_enabled:
            kept_indices = {int(position) for position in keep}
            prune_rounds.append({
                "round": round_index,
                "candidates_before": int(index.count),
                "candidates_after": len(kept_indices),
                "states_before": [_state_record(state) for state in index.states],
                "states_kept": [
                    _state_record(state) for position, state in enumerate(index.states)
                    if position in kept_indices
                ],
                "states_dropped": [
                    _state_record(state) for position, state in enumerate(index.states)
                    if position not in kept_indices
                ],
                "deciding_excluded_proposals": bool(stereo_cells),
                "support_prior_applied": False,
            })
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
            if trace_enabled and prune_rounds:
                prune_rounds[-1]["support_prior_applied"] = True
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
    cost_records: list[dict[str, object]] | None = None
    aggregation_records: list[dict[str, object]] | None = None
    if trace_enabled:
        # Taken after the light/exposure/self-calibration/pruning passes, from the
        # same observations and view weights `_accumulate_costs` just consumed, so
        # the records describe the final scored pass and no earlier one.
        cost_records = _observation_cost_records(
            observations, index, view_weights, trace_observations, trace_best)
        aggregation_records = _aggregation_records(totals, weights)
    observed_frames: dict[BlockCoordinate, set[str]] = {}
    for frame_id, coordinate, _costs, weight in observations:
        if weight * view_weights.get(frame_id, 1.0) > 0.0:
            observed_frames.setdefault(coordinate, set()).add(frame_id)
    # Trace-only side table: the cheapest-first order kept as deep as the trace
    # limit reaches. Empty when no trace was requested, and read by nothing below
    # except the trace writer at the end of this function.
    ranked_candidates: dict[BlockCoordinate, list[tuple[BlockState, float]]] = {}

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
        # Trace-only: the same cheapest-first cost order, kept deeper than the
        # production top-K map so an evaluator can read availability at K = 16/32
        # as well as 8. Deriving it from the same ``mean`` costs and the same
        # ``_top_candidates`` helper keeps the order identical to production; the
        # winner and every later decision stay exactly the unchanged map's.
        if trace_enabled:
            ranked_candidates[coordinate] = sorted(
                _top_candidates(mean, index, keep=stage_trace_candidate_limit).items(),
                key=lambda item: (-item[1], item[0].name),
            )
        state = max(candidate_scores, key=candidate_scores.get)
        merged[coordinate] = (
            state,
            confidence,
            observed_frames.get(coordinate, set()),
            [state],
            candidate_scores,
        )

    raw_candidates: dict[BlockCoordinate, dict[str, object]] | None = None
    if trace_enabled:
        # The raw pre-smoothing ranking, taken before adjacency mutates the
        # candidate maps. ``candidate_winner`` is the argmax the un-smoothed
        # pipeline would have placed, in state-property form. ``top_candidates``
        # is the production top-K map; ``ranked_candidates`` is the same
        # cheapest-first order recorded as deep as the trace limit reaches.
        raw_candidates = {}
        for coordinate, values in merged.items():
            candidate_scores = values[4]
            if not candidate_scores:
                continue
            ranked = sorted(candidate_scores.items(), key=lambda item: (-item[1], item[0].name))
            deep_ranked = ranked_candidates.get(coordinate) or ranked
            raw_candidates[coordinate] = {
                "raw_winner": _state_record(ranked[0][0]),
                "top_candidates": [
                    {"state": _state_record(state), "cost": -float(score)}
                    for state, score in ranked
                ],
                "ranked_candidates": [
                    {"state": _state_record(state), "cost": -float(score)}
                    for state, score in deep_ranked
                ],
                "candidate_count": len(deep_ranked),
                # ``>=`` because the writer cannot see past its own limit: a list
                # that exactly fills the limit may or may not continue, so it is
                # reported conservatively as possibly truncated.
                "top_k_truncated": len(deep_ranked) >= stage_trace_candidate_limit,
            }

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
            settled, scene, cameras, frames, fitting, limit=proposal_limit,
            usable_masks=usable_pixels)
        fitted = fit_states(
            list(wanted), settled, scene, cameras, frames, fitting,
            usable_masks=usable_pixels,
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
    if trace_enabled:
        _emit_stage_trace(
            stage_trace, palette=palette, index=index, raw=raw,
            prune_rounds=prune_rounds, raw_candidates=raw_candidates, merged=merged,
            volume=volume, candidate_limit=stage_trace_candidate_limit,
            observation_records=trace_observations,
            observation_cost_records=cost_records,
            aggregation_records=aggregation_records,
        )
        if stage_sidecar is not None and sidecar_records is not None:
            # The opt-in payload sidecar, filled from the same run's captured
            # resolved-face arrays. It is written to its own destination by the
            # caller; nothing here reads it back.
            _emit_stage_sidecar(stage_sidecar, sidecar_records, trace_observations)
    return volume


def _state_record(state: BlockState) -> dict[str, object]:
    """One block state as a JSON-safe primitive record.

    The properties tuple is stored as an ordered list of pairs so the exact
    state (``dark_oak_log[axis=z]`` and its siblings) survives ``json.dumps``
    without depending on :class:`BlockState` equality's default stripping.
    """
    return {"name": state.name, "properties": [list(pair) for pair in state.properties]}


def _evidence_records(raw: list) -> list[dict[str, object]]:
    """Per-coordinate appearance provenance, reference-agnostic.

    For every coordinate any camera had evidence about, records where that
    evidence came from: how many resolved-face samples and centre-patch fallback
    samples it has, which world face slots and frames contributed, how many
    carry a texture pattern, and the observable area/weight/colour totals. This
    is the input to the classifier described purely in terms of the reconstruction's
    own sampling, so an evaluator can later ask which stage lost a cell.
    """
    grouped: dict[BlockCoordinate, dict[str, object]] = {}
    for frame_id, coordinate, rgb, _descriptor, slot, weight, pattern, area in raw:
        entry = grouped.get(coordinate)
        if entry is None:
            entry = {
                "coordinate": [int(value) for value in coordinate],
                "observation_count": 0,
                "resolved_samples": 0,
                "fallback_samples": 0,
                "pattern_samples": 0,
                "slots": set(),
                "frames": set(),
                "slots_resolved": set(),
                "slots_fallback": set(),
                "resolved_area": 0.0,
                "total_weight": 0.0,
                "rgb_sum": [0.0, 0.0, 0.0],
            }
            grouped[coordinate] = entry
        entry["observation_count"] = int(entry["observation_count"]) + 1  # type: ignore[arg-type]
        entry["slots"].add(str(slot))  # type: ignore[union-attr]
        entry["frames"].add(str(frame_id))  # type: ignore[union-attr]
        if float(area) > 0.0:
            entry["resolved_samples"] = int(entry["resolved_samples"]) + 1  # type: ignore[arg-type]
            entry["slots_resolved"].add(str(slot))  # type: ignore[union-attr]
            entry["resolved_area"] = float(entry["resolved_area"]) + float(area)  # type: ignore[arg-type]
        else:
            entry["fallback_samples"] = int(entry["fallback_samples"]) + 1  # type: ignore[arg-type]
            entry["slots_fallback"].add(str(slot))  # type: ignore[union-attr]
        if pattern is not None:
            entry["pattern_samples"] = int(entry["pattern_samples"]) + 1  # type: ignore[arg-type]
        entry["total_weight"] = float(entry["total_weight"]) + float(weight)  # type: ignore[arg-type]
        totals = entry["rgb_sum"]  # type: ignore[assignment]
        for channel in range(3):
            totals[channel] += float(np.asarray(rgb, dtype=float)[channel])  # type: ignore[index]
    records: list[dict[str, object]] = []
    for coordinate in sorted(grouped):
        entry = grouped[coordinate]
        observations = max(int(entry["observation_count"]), 1)  # type: ignore[arg-type]
        records.append({
            "coordinate": entry["coordinate"],
            "observation_count": entry["observation_count"],
            "resolved_samples": entry["resolved_samples"],
            "fallback_samples": entry["fallback_samples"],
            "pattern_samples": entry["pattern_samples"],
            "has_resolved_face": int(entry["resolved_samples"]) > 0,  # type: ignore[arg-type]
            "slots": sorted(entry["slots"]),  # type: ignore[arg-type]
            "frames": sorted(entry["frames"]),  # type: ignore[arg-type]
            "slots_resolved": sorted(entry["slots_resolved"]),  # type: ignore[arg-type]
            "slots_fallback": sorted(entry["slots_fallback"]),  # type: ignore[arg-type]
            "resolved_area": round(float(entry["resolved_area"]), 3),  # type: ignore[arg-type]
            "total_weight": round(float(entry["total_weight"]), 3),  # type: ignore[arg-type]
            "mean_rgb": [
                round(float(channel) / observations, 3)
                for channel in entry["rgb_sum"]  # type: ignore[union-attr]
            ],
        })
    return records


def _array_digest(value: object) -> str | None:
    """SHA-256 over an array-like's shape, dtype and raw bytes.

    ``None`` stays ``None`` so a centre-patch fallback - which has no rectified
    patch, validity mask or coverage - records an explicit null rather than a
    digest of nothing. Only a non-contiguous input is copied, and only to read
    its bytes, so the caller's array is never mutated.
    """
    if value is None:
        return None
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(f"{array.shape}|{array.dtype.str}|".encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _array_payload(value: np.ndarray) -> dict[str, object]:
    """One array as an exact, self-describing JSON-safe payload record.

    ``dtype``/``shape`` freeze the array's domain and ``base64`` is its exact
    C-order bytes in that dtype, so a reader can restore the float32/bool array
    losslessly (and re-derive the same shape/dtype-bound digest). Only a
    non-contiguous input is copied, and only to read its bytes, so the caller's
    array is never mutated.
    """
    array = np.ascontiguousarray(np.asarray(value))
    return {
        "dtype": array.dtype.str,
        "shape": [int(size) for size in array.shape],
        "base64": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
    }


def _observation_record(
    *,
    ordinal: int,
    frame_id: str,
    coordinate: BlockCoordinate,
    slot: str,
    source: str,
    area: float,
    depth: float,
    visible: float | None,
    facing: float | None,
    weight: float,
    observed_patch: np.ndarray | None,
    valid: np.ndarray | None,
    coverage: np.ndarray | None,
    rgb: np.ndarray,
    descriptor: object,
    pattern: np.ndarray | None,
) -> dict[str, object]:
    """One retained raw observation as a JSON-safe diagnostic record.

    ``weight`` is the pre-correction weight the sampler produced; ``rgb``,
    ``descriptor`` and ``pattern`` are the pre-correction values that entered
    ``raw``. The digests are shape/dtype-bound, so a replay can prove it
    reconstructed the same inputs without the trace carrying any pixels.
    """
    return {
        "ordinal": int(ordinal),
        "frame_id": str(frame_id),
        "coordinate": [int(value) for value in coordinate],
        "slot": str(slot),
        "source": str(source),
        "area": float(area),
        "depth": float(depth),
        "visible": None if visible is None else float(visible),
        "facing": None if facing is None else float(facing),
        "weight": float(weight),
        "observed_patch": _array_digest(observed_patch),
        "valid": _array_digest(valid),
        "coverage": _array_digest(coverage),
        "rgb": _array_digest(rgb),
        "descriptor": _array_digest(descriptor),
        "pattern": _array_digest(pattern),
    }


def _sidecar_record(
    *,
    ordinal: int,
    frame_id: str,
    coordinate: BlockCoordinate,
    slot: str,
    area: float,
    depth: float,
    visible: float,
    facing: float,
    weight: float,
    patch: np.ndarray,
    observed_patch: np.ndarray,
    valid: np.ndarray,
    coverage: np.ndarray,
) -> dict[str, object]:
    """One retained resolved-face observation's actual sampled payload.

    Two pixel arrays are carried because they are not the same array. ``patch``
    is the production sampler's raw rectified patch - the array E1 retains and
    the authority for exact raw-patch parity. ``observed_patch`` is the
    scorer-facing array the E2A trace digests, with invalid pixels replaced by the
    coverage-weighted ``rgb``; its payload and digest stay tied to that trace
    observation. ``valid`` and ``coverage`` are the sampler's own masks, and the
    ``digests`` block uses the same shape/dtype-bound SHA-256 as the E2A trace, so
    a record's identity and its observed-patch/mask digests can be checked against
    the trace observation at the same ordinal. The raw-patch digest has no trace
    counterpart by design. No reference label, evaluator truth or reconstruction
    result is read or recorded here, and no array is mutated.
    """
    return {
        "ordinal": int(ordinal),
        "frame_id": str(frame_id),
        "coordinate": [int(value) for value in coordinate],
        "slot": str(slot),
        "source": "resolved_face",
        "area": float(area),
        "depth": float(depth),
        "visible": float(visible),
        "facing": float(facing),
        "weight": float(weight),
        "raw_patch": _array_payload(patch),
        "observed_patch": _array_payload(observed_patch),
        "valid": _array_payload(valid),
        "coverage": _array_payload(coverage),
        "digests": {
            "raw_patch": _array_digest(patch),
            "observed_patch": _array_digest(observed_patch),
            "valid": _array_digest(valid),
            "coverage": _array_digest(coverage),
        },
    }


def _emit_stage_sidecar(
    stage_sidecar: dict[str, object],
    records: list[dict[str, object]],
    observation_records: list[dict[str, object]],
) -> None:
    """Fill a caller-supplied mapping with the resolved-face payload sidecar.

    Kept separate from the E2A trace: it carries the sampled pixel bytes the trace
    deliberately omits - both the raw sampler patch and the scorer-facing observed
    patch - for its own opt-in path. A centre-patch fallback has no rectified face
    and therefore no payload, so it is excluded from ``records`` and counted
    explicitly; the header also records the full observation count so a reader can
    reconcile the sidecar against the trace. Written once, at the end, and read
    back by nothing in this module.
    """
    fallbacks = sum(
        1 for record in observation_records
        if str(record.get("source")) == "centre_patch_fallback"
    )
    stage_sidecar["schema"] = STAGE_SIDECAR_SCHEMA
    stage_sidecar["trace_schema"] = 2
    stage_sidecar["observation_count"] = len(observation_records)
    stage_sidecar["resolved_face_records"] = len(records)
    stage_sidecar["excluded_fallback_observations"] = fallbacks
    stage_sidecar["records"] = records


def _observation_cost_records(
    observations: list[tuple[str, BlockCoordinate, np.ndarray, float]],
    index: PaletteIndex,
    view_weights: Mapping[str, float],
    observation_records: list[dict[str, object]],
    best_costs: list[float],
) -> list[dict[str, object]]:
    """One record per observation from the run's final scored pass.

    ``palette_order`` is the final ``index``'s state order, so the cost vector
    is interpretable on its own. ``weight`` is the post-fit weight the scorer
    produced, ``view_weight`` is the per-frame trust, and ``effective_weight``
    is their product - exactly the quantity ``_accumulate_costs`` sums with.
    """
    palette_order = [_state_record(state) for state in index.states]
    records: list[dict[str, object]] = []
    for ordinal, (frame_id, coordinate, costs, weight) in enumerate(observations):
        base = observation_records[ordinal]
        view_weight = float(view_weights.get(frame_id, 1.0))
        records.append({
            "ordinal": int(ordinal),
            "frame_id": str(frame_id),
            "coordinate": [int(value) for value in coordinate],
            "slot": base["slot"],
            "source": base["source"],
            "palette_order": palette_order,
            "costs": [float(value) for value in costs],
            "best_cost": float(best_costs[ordinal]),
            "weight": float(weight),
            "view_weight": view_weight,
            "effective_weight": float(weight) * view_weight,
        })
    return records


def _aggregation_records(
    totals: dict[BlockCoordinate, np.ndarray],
    weights: dict[BlockCoordinate, float],
) -> list[dict[str, object]]:
    """Exact ``_accumulate_costs`` totals and weights in sorted coordinate order."""
    return [
        {
            "coordinate": [int(value) for value in coordinate],
            "total": [float(value) for value in totals[coordinate]],
            "weight": float(weights[coordinate]),
        }
        for coordinate in sorted(totals)
    ]


def _emit_stage_trace(
    stage_trace: dict[str, object],
    *,
    palette: tuple[PaletteColor, ...],
    index: PaletteIndex,
    raw: list,
    prune_rounds: list[dict[str, object]],
    raw_candidates: dict[BlockCoordinate, dict[str, object]] | None,
    merged: dict,
    volume: BlockVolume,
    candidate_limit: int,
    observation_records: list[dict[str, object]] | None = None,
    observation_cost_records: list[dict[str, object]] | None = None,
    aggregation_records: list[dict[str, object]] | None = None,
) -> None:
    """Fill a caller-supplied mapping with the run's stagewise, reference-free facts.

    Written once, at the end, and read back by nothing in this module. The two
    coordinate-keyed tables are emitted as sorted JSON-safe lists, and every
    value is a primitive so the whole trace survives ``json.dumps`` as-is.
    """
    round_states: list[dict[str, set[str]]] = []
    for record in prune_rounds:
        round_states.append({
            "kept": {str(state.get("name")) for state in record["states_kept"]},  # type: ignore[union-attr]
            "dropped": {str(state.get("name")) for state in record["states_dropped"]},  # type: ignore[union-attr]
        })

    coordinate_traces: list[dict[str, object]] = []
    coordinates = sorted(set(raw_candidates or {}) | {c for c in merged} | set(volume.blocks))
    for coordinate in coordinates:
        entry: dict[str, object] = {"coordinate": [int(value) for value in coordinate]}
        candidates = (raw_candidates or {}).get(coordinate)
        if candidates is not None:
            entry.update(candidates)
        values = merged.get(coordinate)
        if values is not None:
            chosen = _choose_state_with_local_family_support(
                coordinate, values[4], merged, {item.state: item.family for item in palette}
            )
            settled = max(values[4], key=values[4].get) if values[4] else None
            entry["post_smoothing_label"] = (
                _state_record(settled) if settled is not None else None
            )
            entry["selected_state"] = _state_record(chosen) if chosen is not None else None
        placed = volume.blocks.get(coordinate)
        entry["final_state"] = (
            _state_record(placed.state) if placed is not None else None
        )
        if round_states and candidates is not None:
            # Per-coordinate provenance for the *global* pruning rounds: which of
            # this cell's own ranked candidates each round kept, and which it
            # dropped. The reference state is not known here, so the trace never
            # privileges one winner - the evaluator tests the transformed
            # reference state against the per-round membership in
            # ``support_rounds``/``final_palette`` after reconstruction finishes,
            # and reads these markers only as per-cell richness.
            ranked_names = [
                str(candidate["state"]["name"])
                for candidate in candidates.get("top_candidates", [])
                if isinstance(candidate, dict) and isinstance(candidate.get("state"), dict)
            ]
            entry["support"] = [
                {
                    "round": int(record["round"]),
                    "candidates_before": int(record["candidates_before"]),
                    "candidates_after": int(record["candidates_after"]),
                    "candidate_states_kept": [
                        name for name in ranked_names
                        if name in round_states[position]["kept"]
                    ],
                    "candidate_states_dropped": [
                        name for name in ranked_names
                        if name not in round_states[position]["kept"]
                    ],
                }
                for position, record in enumerate(prune_rounds)
            ]
        coordinate_traces.append(entry)

    stage_trace["schema"] = 2
    stage_trace["candidate_limit"] = int(candidate_limit)
    stage_trace["palette"] = {
        "size": len(palette),
        "states": [_state_record(item.state) for item in palette],
        "families": {
            item.state.name: item.family for item in palette
        },
    }
    stage_trace["support_rounds"] = [
        {
            "round": int(record["round"]),
            "candidates_before": int(record["candidates_before"]),
            "candidates_after": int(record["candidates_after"]),
            "states_before": record["states_before"],
            "states_kept": record["states_kept"],
            "states_dropped": record["states_dropped"],
            "deciding_excluded_proposals": bool(record["deciding_excluded_proposals"]),
            "support_prior_applied": bool(record["support_prior_applied"]),
        }
        for record in prune_rounds
    ]
    stage_trace["final_palette"] = {
        "size": int(index.count),
        "states": [_state_record(state) for state in index.states],
    }
    stage_trace["coordinates"] = coordinate_traces
    # Schema-2 additions, diagnostic only. Every schema-1 key above keeps its
    # exact meaning; these three are read back by nothing in production.
    stage_trace["observations"] = observation_records or []
    stage_trace["observation_costs"] = observation_cost_records or []
    stage_trace["aggregation"] = aggregation_records or []


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


