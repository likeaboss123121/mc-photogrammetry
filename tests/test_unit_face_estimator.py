"""Synthetic arrays/generated archive only; no evaluator or real game assets."""
from dataclasses import FrozenInstanceError, replace
from copy import deepcopy
import io
import json
import zipfile

import numpy as np
from PIL import Image
import pytest

from mc_photogrammetry.blocks.unit_face_catalogue import load_unit_face_catalogue
from mc_photogrammetry.reconstruction.calibration import CalibrationResult
from mc_photogrammetry.reconstruction import lattice
from mc_photogrammetry.reconstruction import unit_face_estimator as fit
from mc_photogrammetry.reconstruction import unit_face_lattice as taps


CONDITIONING = 'a' * 64
SLOTS = ('down', 'up', 'north', 'south', 'east', 'west')
# Preregistered asymmetric texture, independent of all fitting observations.
PATTERN = np.random.default_rng(1827).integers(20, 236, (4, 4, 3), dtype=np.uint8)


def catalogue(pattern=PATTERN, *, aliases=1, turns=(0,), unsupported=False, rival_pattern=None,
              variant=(0, 0)):
    rgba = np.concatenate((pattern, np.full(pattern.shape[:2] + (1,), 255, np.uint8)), axis=2)
    png = io.BytesIO()
    Image.fromarray(rgba).save(png, format='PNG')
    assets = {'assets/minecraft/textures/block/pattern.png': png.getvalue()}
    for turn in turns:
        model = {'textures': {'all': 'block/pattern'}, 'elements': [{
            'from': [0, 0, 0], 'to': [8 if unsupported else 16, 16, 16],
            'faces': {slot: {'texture': '#all', 'rotation': turn} for slot in SLOTS}}]}
        assets[f'assets/minecraft/models/block/cube{turn}.json'] = model
        for alias in range(aliases):
            assets[f'assets/minecraft/blockstates/state{turn}_{alias}.json'] = {
                'variants': {'': {'model': f'block/cube{turn}', 'x': variant[0], 'y': variant[1]}}}
    if rival_pattern is not None:
        rival_rgba = np.concatenate((rival_pattern, np.full(rival_pattern.shape[:2]+(1,), 255, np.uint8)), axis=2)
        rival_png = io.BytesIO()
        Image.fromarray(rival_rgba).save(rival_png, format='PNG')
        assets['assets/minecraft/textures/block/rival.png'] = rival_png.getvalue()
        assets['assets/minecraft/models/block/rival.json'] = {
            'textures': {'all': 'block/rival'}, 'elements': [{'from': [0, 0, 0], 'to': [16, 16, 16],
                'faces': {slot: {'texture': '#all'} for slot in SLOTS}}]}
        assets['assets/minecraft/blockstates/rival.json'] = {'variants': {'': {'model': 'block/rival'}}}
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        for name, value in sorted(assets.items()):
            archive.writestr(zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0)),
                             value if isinstance(value, bytes) else json.dumps(value))
    return load_unit_face_catalogue(stream.getvalue())


def context(shape=(112, 128), focal=110, yaw=0, pitch=0, roll=0, centre=None, axis=2):
    h, w = shape
    calibration = CalibrationResult(focal, yaw, pitch, w, h, 90, 100, 0.1,
                                     (30, 30, 30), roll, centre)
    return fit.context_from_calibration(calibration, axis, conditioning_sha256=CONDITIONING)


def render(ctx, pattern=PATTERN, distance=3.23, phase=(0.173, 0.287), turn=0,
           gain=(1.01, 0.99, 1.02), bias=(0.003, -0.004, 0.002), slot='north', variant=(0, 0)):
    """Independent source-ray renderer with explicit north-face UV rotations.

    Uses native texture bilinear values and nearest-byte sensor quantization.
    No estimator sampling/coordinate/exposure routine is called.
    """
    h, w = ctx.source_shape
    yy, xx = np.indices((h, w))
    ray = np.stack(((xx - ctx.principal_point[0]) / ctx.focal_pixels,
                    -(yy - ctx.principal_point[1]) / ctx.focal_pixels,
                    np.ones((h, w))), axis=-1)
    axes = np.asarray(ctx.axis_matrix)
    denom = np.abs(ray @ axes[ctx.axis])
    point = ray / denom[..., None]
    others = [i for i in range(3) if i != ctx.axis]
    bx = distance * (point @ axes[others[0]]) + phase[0]
    by = distance * (point @ axes[others[1]]) + phase[1]
    # Literal accepted north raw corner correspondences, no corner-index formula.
    u0, v0 = {'north': (-bx+.5, -by+.5), 'south': (bx+.5, -by+.5),
              'west': (by+.5, -bx+.5), 'east': (-by+.5, -bx+.5),
              'up': (bx+.5, by+.5), 'down': (bx+.5, -by+.5)}[slot]
    if variant == (90, 90):
        # Independently stated x/y composition (x,y,z)->(y,z,x): source
        # north becomes down. Its raw U=-sourceX=-worldZ, V=-sourceY=-worldX.
        assert slot == 'down'
        u0, v0 = -by+.5, -bx+.5
    else:
        assert variant == (0, 0)
    u, v = {0: (u0, v0), 90: (v0, 1-u0), 180: (1-u0, 1-v0), 270: (1-v0, u0)}[turn]
    side = pattern.shape[0]
    tx, ty = (u % 1) * side - .5, (v % 1) * side - .5
    ix, iy = np.floor(tx).astype(int), np.floor(ty).astype(int)
    dx, dy = tx - ix, ty - iy
    color = np.zeros((h, w, 3))
    for ox, oy, weight in ((0, 0, (1-dx)*(1-dy)), (1, 0, dx*(1-dy)),
                           (0, 1, (1-dx)*dy), (1, 1, dx*dy)):
        color += weight[..., None] * pattern[(iy+oy) % side, (ix+ox) % side] / 255
    return np.rint(np.clip(color * gain + bias, 0, 1) * 255).astype(np.uint8)


def support(image, ctx, samples=16, *, occlusion=False):
    h, w = image.shape[:2]
    yy, xx = np.indices((h, w))
    usable = (xx > 4) & (xx < w-5) & (yy > 4) & (yy < h-5)
    if occlusion:
        usable[(xx > w*.4) & (xx < w*.6)] = False
    generation = usable & ((xx // 8 + yy // 8) % 2 == 0)
    validation = usable & ~generation
    partition = taps.SourceTapPartition.from_masks(generation, validation, usable,
        np.ones_like(usable), conditioning_sha256=CONDITIONING)
    witnesses = []
    lattice.rectify_plane(image, ctx.axis, np.asarray(ctx.axis_matrix), ctx.focal_pixels,
        extent=.42, samples=samples, principal_point=ctx.principal_point,
        side=1 if ctx.axis_matrix[ctx.axis][2] >= 0 else -1, witness_sink=witnesses)
    witness = witnesses[0]
    groups = []
    for index in range(witness.map_x.size):
        first, last = witness.tap_ptr[index:index+2]
        pixels = witness.tap_source_indices[first:last]
        if not witness.depth_valid.flat[index] or np.any(pixels < 0):
            continue
        roles = {partition.roles[int(p)] for p in pixels}
        if roles in ({1}, {2}):
            role = 'generation' if roles == {1} else 'validation'
            groups.append(taps.DependencyGroup(f'{role}-{index}', role, ((0, index),)))
    return taps.partition_tap_support(partition, witnesses, groups)


def policy(**changes):
    return replace(fit.SearchPolicy(distance_bounds=(1.4, 6.8)), **changes)


def circle_error(a, b):
    return abs((a - b + .5) % 1 - .5)


def test_context_consumes_calibration_without_distance_or_positions():
    ctx = context(yaw=11, pitch=7, roll=3, centre=(61.2, 50.8))
    assert ctx.source_shape == (112, 128)
    assert ctx.principal_point == (61.2, 50.8)
    assert np.allclose(np.asarray(ctx.axis_matrix) @ np.asarray(ctx.axis_matrix).T, np.eye(3))
    assert len(ctx.sha256) == 64
    with pytest.raises(FrozenInstanceError):
        ctx.focal_pixels = 4
    with pytest.raises(ValueError, match='invalid_plane_context'):
        replace(ctx, axis_matrix=((1., 0., 0.),) * 3)


def test_exact_box_exposure_covers_boundary_and_constant_model():
    rng = np.random.default_rng(5903)  # disjoint numerical plumbing set
    model = rng.random((64, 3))
    observed = model * (1.08, .9, 1.01) + (.03, -.04, .002)
    error, gain, bias = fit._exposure(model, observed, policy())
    assert gain[0] == 1.05 and gain[1] == .95
    assert bias[0] == .02 and bias[1] == -.02
    assert gain[2] == pytest.approx(1.01)
    assert bias[2] == pytest.approx(.002)
    assert error > 0
    error, gain, bias = fit._exposure(np.ones((64, 3))*.4, np.ones((64, 3))*.4, policy())
    assert error == pytest.approx(0, abs=1e-14)


@pytest.mark.parametrize('turn', [0, 90, 180, 270])
def test_native_raw_corner_prediction_matches_independent_renderer(turn):
    ctx = context()
    cat = catalogue(turns=(turn,))
    face = next(f for f in cat.faces if f.face_slot == 'north')
    texture, transform, geometry = fit._template(face, 2)
    image = render(ctx, turn=turn, gain=(1, 1, 1), bias=(0, 0, 0))
    indices = tuple(range(1000, 3000, 7))
    coords, _, _ = fit._rays(ctx, indices)
    prediction = fit._sample(texture, (coords * 3.23 + (.173, .287)) @ transform.T + .5)
    assert np.max(abs(prediction - image.reshape(-1, 3)[list(indices)] / 255)) <= .5/255 + 1e-12
    assert len(geometry) == 64


def test_real_rectification_witness_support_has_disjoint_source_roles():
    ctx = context()
    image = render(ctx)
    proof = support(image, ctx)
    assert proof.status == 'complete'
    assert len(proof.generation_pixels) >= 48 and len(proof.validation_pixels) >= 48
    assert not set(proof.generation_pixels) & set(proof.validation_pixels)
    assert all(s.witness_sha256 == proof.witnesses[s.witness_index].witness_sha256 for s in proof.samples)


def test_identifiable_baseline_numeric_distance_and_both_phases():
    ctx = context()
    image = render(ctx)
    output = fit.estimate_unit_face(image, ctx, support(image, ctx), catalogue(), policy())
    assert output.status == 'supported', (output.reason, output.work_used, output.distance_interval,
                                           output.phase_intervals, output.candidates[:1])
    winner = output.representative
    assert abs(winner.distance/3.23-1) < .05
    assert circle_error(winner.phase[0], .173) < .1
    assert circle_error(winner.phase[1], .287) < .1
    assert output.distance_interval[0] <= 3.23 <= output.distance_interval[1]
    assert all((truth-start) % 1 <= width for (start, width), truth in
               zip(output.phase_intervals, (.173, .287)))
    assert output.search_complete and output.work_used <= output.work_budget
    assert not output.placement_supported
    assert winner.validation_rmse < .03
    json.dumps(output.to_record(), allow_nan=False)


@pytest.mark.parametrize('shape,focal,yaw,pitch,roll,centre,distance,phase', [
    ((80, 96), 82, 0, 0, 0, None, 2.17, (.93, .04)),
    ((128, 160), 150, 18, -11, 2.7, (69.4, 57.2), 4.61, (.371, .623)),
    ((112, 128), 120, -16, 13, -2.1, (72.1, 48.9), 3.71, (.713, .119)),
    ((144, 176), 174, 9, 6, 1.3, None, 6.09, (.241, .813)),
])
def test_synthetic_resolution_crop_optics_pose_and_distance_matrix(
        shape, focal, yaw, pitch, roll, centre, distance, phase):
    ctx = context(shape, focal, yaw, pitch, roll, centre)
    image = render(ctx, distance=distance, phase=phase)
    output = fit.estimate_unit_face(image, ctx, support(image, ctx), catalogue(), policy())
    assert output.status == 'supported', (output.reason, output.work_used, output.distance_interval,
                                           output.phase_intervals)
    assert abs(output.representative.distance/distance-1) < .05
    assert all(circle_error(a, b) < .1 for a, b in zip(output.representative.phase, phase))
    assert output.distance_interval[0] <= distance <= output.distance_interval[1]
    assert all((truth-start) % 1 <= width for (start, width), truth in zip(output.phase_intervals, phase))
    assert any(box.distance[0] <= distance <= box.distance[1]
        and box.phase_u[0] <= phase[0] <= box.phase_u[1]
        and box.phase_v[0] <= phase[1] <= box.phase_v[1] for box in output.cover)


def test_all_geometry_distinct_uv_orientations_and_equal_geometry_state_aliases():
    ctx = context()
    image = render(ctx, turn=90)
    cat = catalogue(aliases=2, turns=(0, 90, 180, 270))
    output = fit.estimate_unit_face(image, ctx, support(image, ctx), cat, policy())
    assert output.status == 'supported', (output.reason, output.work_used)
    assert len(output.candidates) == 4  # no appearance/identity rival discarded
    assert len({c.geometry_sha256 for c in output.candidates}) == 4
    assert all(len(c.state_aliases) == 2 for c in output.candidates)
    assert output.candidates[0].face_sha256 in {f.sha256 for f in cat.faces if f.uv_rotation == 90}


def test_heldout_mutation_changes_veto_only_generation_rank_and_fit_are_frozen():
    ctx = context()
    image = render(ctx)
    proof = support(image, ctx)
    cat = catalogue(turns=(0, 90))
    before = fit.estimate_unit_face(image, ctx, proof, cat, policy())
    changed = image.copy()
    changed.reshape(-1, 3)[list(proof.validation_pixels)] = np.random.default_rng(4781).integers(
        0, 256, (len(proof.validation_pixels), 3), dtype=np.uint8)
    after = fit.estimate_unit_face(changed, ctx, proof, cat, policy())
    assert before.status == 'supported'
    assert after.status == 'abstained' and after.reason == 'validation_residual_failure'
    assert before.source_generation_sha256 == after.source_generation_sha256
    assert before.source_validation_sha256 != after.source_validation_sha256
    assert before.distance_interval == after.distance_interval
    assert before.phase_intervals == after.phase_intervals
    assert before.work_used == after.work_used
    assert replace(before.representative, validation_rmse=None, validation_max_tile_rmse=None) == replace(
        after.representative, validation_rmse=None, validation_max_tile_rmse=None)
    for first, second in zip(before.candidates, after.candidates):
        assert replace(first, validation_rmse=None, validation_max_tile_rmse=None) == replace(
            second, validation_rmse=None, validation_max_tile_rmse=None)


def test_excluded_gui_pixels_cannot_change_any_frozen_context_fit_evidence():
    ctx = context()
    image = render(ctx)
    proof = support(image, ctx)
    before = fit.estimate_unit_face(image, ctx, proof, catalogue(), policy())
    changed = image.copy()
    owned = set(proof.generation_pixels) | set(proof.validation_pixels)
    excluded = [i for i in range(image.shape[0]*image.shape[1]) if i not in owned]
    changed.reshape(-1, 3)[excluded] = 255 - changed.reshape(-1, 3)[excluded]
    after = fit.estimate_unit_face(changed, ctx, proof, catalogue(), policy())
    assert before.status == 'supported'
    assert before == after


def test_two_rectified_sampling_densities_agree_within_enclosed_uncertainty():
    ctx = context()
    image = render(ctx)
    outputs = [fit.estimate_unit_face(image, ctx, support(image, ctx, samples=n), catalogue(), policy())
               for n in (12, 18)]
    assert all(o.status == 'supported' for o in outputs), [o.reason for o in outputs]
    first, second = outputs
    assert first.support_sha256 != second.support_sha256
    assert max(first.distance_interval[0], second.distance_interval[0]) <= min(
        first.distance_interval[1], second.distance_interval[1])
    assert abs(first.candidates[0].distance/second.candidates[0].distance-1) < .05
    assert all(circle_error(a, b) < .1 for a, b in zip(first.candidates[0].phase, second.candidates[0].phase))


@pytest.mark.parametrize('kind', ['flat', 'repeated', 'two_texel_harmonic', 'symmetric'])
def test_flat_symmetric_repeated_and_two_texel_harmonic_refuse_unique_geometry(kind):
    if kind == 'flat':
        pattern = np.full((4, 4, 3), 90, np.uint8)
    elif kind == 'repeated':
        pattern = np.tile(PATTERN[:2, :2], (2, 2, 1))
    elif kind == 'two_texel_harmonic':
        pattern = np.asarray([[[30, 80, 180], [210, 160, 60]],
                              [[210, 160, 60], [30, 80, 180]]], np.uint8)
    else:
        pattern = np.array(PATTERN)
        for y in range(4):
            for x in range(4):
                pattern[y, x] = PATTERN[min(y, 3-y), min(x, 3-x)]
    ctx = context()
    image = render(ctx, pattern)
    cat = catalogue(pattern, turns=(0, 90, 180, 270) if kind == 'symmetric' else (0,))
    output = fit.estimate_unit_face(image, ctx, support(image, ctx), cat,
                                    policy(max_work=20_000))
    assert output.status == 'abstained'
    assert output.reason in ('flat_or_uninformative_evidence', 'scale_or_phase_ambiguous',
                             'competing_geometry_aliases', 'search_work_exhausted')
    assert not output.placement_supported


@pytest.mark.parametrize('kind', ['mixed_material', 'mixed_depth', 'occluded'])
def test_mixed_material_depth_and_occluded_pixels_cannot_claim_confident_fit(kind):
    ctx = context()
    image = render(ctx)
    if kind == 'mixed_depth':
        other = render(ctx, distance=5.1)
    elif kind == 'mixed_material':
        other = render(ctx, 255-PATTERN)
    else:
        other = np.zeros_like(image)
    image[:, image.shape[1]//2:] = other[:, image.shape[1]//2:]
    output = fit.estimate_unit_face(image, ctx, support(image, ctx), catalogue(), policy(max_work=10_000))
    assert output.status == 'abstained'
    assert output.reason in ('generation_residual_failure', 'validation_residual_failure',
                             'search_work_exhausted', 'scale_or_phase_ambiguous')


def test_explicit_source_tap_overlap_and_depth_refusals_are_atomic():
    ctx = context()
    image = render(ctx)
    proof = support(image, ctx)
    group = next(g for g in proof.groups if g.role == 'generation')
    overlapping = taps.DependencyGroup('overlap', 'validation', group.sample_refs)
    witnesses = [lattice.rectification_witness_v1(
        np.frombuffer(w.map_x, np.float32).reshape(w.map_shape),
        np.frombuffer(w.map_y, np.float32).reshape(w.map_shape),
        np.frombuffer(w.depth_valid, np.bool_).reshape(w.map_shape), source_shape=w.source_shape)
        for w in proof.witnesses]
    refusal = taps.partition_tap_support(proof.partition, witnesses, proof.groups + (overlapping,))
    assert refusal.status == 'refused' and refusal.reason == 'overlapping_source_taps'
    output = fit.estimate_unit_face(image, ctx, refusal, catalogue(), policy())
    assert output.status == 'abstained' and output.reason == 'source_support_unavailable'
    assert output.candidates == () and output.work_used == 0
    depth = np.array(witnesses[0].depth_valid)
    depth.flat[group.sample_refs[0][1]] = False
    invalid = lattice.rectification_witness_v1(witnesses[0].map_x, witnesses[0].map_y, depth,
                                              source_shape=ctx.source_shape)
    refusal = taps.partition_tap_support(proof.partition, (invalid,), proof.groups)
    assert refusal.reason == 'invalid_depth_sample'
    assert fit.estimate_unit_face(image, ctx, refusal, catalogue(), policy()).status == 'abstained'


def test_missing_unsupported_assets_context_mismatch_and_budget_caps():
    ctx = context()
    image = render(ctx)
    proof = support(image, ctx)
    cat = catalogue()
    empty = replace(cat, faces=(), exclusions=(), supported_states=0)
    assert fit.estimate_unit_face(image, ctx, proof, empty, policy()).reason == 'missing_assets'
    assert fit.estimate_unit_face(image, ctx, proof, catalogue(unsupported=True), policy()).reason == 'unsupported_assets'
    assert fit.estimate_unit_face(image, replace(ctx, conditioning_sha256='b'*64), proof,
                                  cat, policy()).reason == 'upstream_conditioning_mismatch'
    cap = fit.estimate_unit_face(image, ctx, proof, cat, policy(max_work=3))
    assert cap.status == 'abstained' and cap.reason == 'search_work_exhausted'
    assert not cap.search_complete and cap.work_used == 2  # one midpoint fit reserved
    assert cap.distance_interval is None and cap.phase_intervals is None
    assert cap.candidates  # a numeric diagnostic never authorizes the incomplete fit
    assert fit.estimate_unit_face(image, ctx, proof, catalogue(turns=(0, 90)),
                                  policy(max_compatible_faces=1)).reason == 'compatible_faces_cap'


def test_native_texture_sampling_resolution_and_grazing_controls():
    ctx = context(focal=32)
    pattern = np.random.default_rng(5714).integers(20, 236, (16, 16, 3), dtype=np.uint8)
    image = render(ctx, pattern)
    output = fit.estimate_unit_face(image, ctx, support(image, ctx), catalogue(pattern),
                                    policy(max_work=1000))
    assert output.status == 'abstained' and output.reason == 'insufficient_resolvable_texels'
    assert output.work_used == 0
    ctx = context(yaw=81)
    image = np.ones(ctx.source_shape + (3,), np.uint8)*80
    proof = support(image, ctx)
    output = fit.estimate_unit_face(image, ctx, proof, catalogue(), policy())
    assert output.status == 'abstained'
    assert output.reason in ('grazing_or_mixed_plane', 'source_support_unavailable',
                             'insufficient_source_evidence', 'flat_or_uninformative_evidence')


def test_conservative_interval_lower_bound_never_exceeds_sampled_box_objectives():
    ctx = context()
    indices = tuple(range(3000, 7000, 31))
    coords, _, _ = fit._rays(ctx, indices)
    observed = render(ctx).reshape(-1, 3)[list(indices)] / 255
    face = next(f for f in catalogue().faces if f.face_slot == 'north')
    texture, matrix, _ = fit._template(face, 2)
    rng = np.random.default_rng(98412)  # disjoint bound controls
    for box in (np.asarray(((1.4, 6.8), (0, 1), (0, 1))),
                np.asarray(((3.2, 3.25), (.15, .20), (.26, .30))),
                np.asarray(((3.0, 3.1), (.8, .9), (.9, 1.0)))):
        lower, _, _ = fit._box_lower(texture, matrix, coords, observed, box, policy())
        for _ in range(40):
            theta = rng.uniform(box[:, 0], box[:, 1])
            model = fit._sample(texture, (coords*theta[0] + theta[1:]) @ matrix.T + .5)
            error, _, _ = fit._exposure(model, observed, policy())
            assert lower <= error + 1e-12


def test_circular_enclosure_uses_full_boxes_including_phase_wrap_and_aliases():
    start, width = fit._circular_enclosure(((0, .02), (.96, 1)))
    assert start == pytest.approx(.96) and width == pytest.approx(.06)
    for point in (0, .01, .02, .96, .97, 1):
        assert (point-start) % 1 <= width + 1e-12
    start, width = fit._circular_enclosure(((.1, .2), (.6, .7)))
    assert width >= .6 - 1e-12  # a half-block alias cannot become a narrow interval


def test_half_double_scale_native_texture_rivals_are_preserved_and_refused():
    ctx = context()
    image = render(ctx)
    cat = catalogue(rival_pattern=np.tile(PATTERN, (2, 2, 1)))
    output = fit.estimate_unit_face(image, ctx, support(image, ctx), cat, policy())
    assert output.status == 'abstained' and output.reason == 'competing_geometry_aliases'
    assert output.search_complete and len(output.candidates) == 2
    distances = sorted(c.distance for c in output.candidates)
    assert distances[0] == pytest.approx(3.23/2, rel=.05)
    assert distances[1] == pytest.approx(3.23, rel=.05)
    assert all(c.generation_rmse < .025 for c in output.candidates)
    assert not output.placement_supported


def test_representative_strict_truth_relative_midpoint_gate_and_boundary_equality():
    phases = ((.95, .1), (.12, .1))
    assert fit._representative_geometry((10., 11.), phases) is None  # exactly 5% at lower endpoint
    assert fit._representative_geometry((95., 105.), phases) is None  # old 10%-of-midpoint rule is unsafe
    point = fit._representative_geometry((10., 10.9), phases)
    assert point[0] == pytest.approx(10.45)
    assert point[1][0] == pytest.approx(0, abs=1e-15)
    for truth in (10., 10.1, 10.9):
        assert abs(point[0]/truth-1) < .05
    assert fit._representative_geometry((10., 10.9), ((0., .2), (.1, .01))) is None
    assert fit._representative_geometry((10., 10.9), ((0., np.nextafter(.2, 0)), (.1, .01))) is None
    assert fit._representative_geometry((10., np.nextafter(11., 0)), phases) is None
    below = 11.
    for _ in range(4):
        below = np.nextafter(below, 0)
    assert fit._representative_geometry((10., below), phases) is not None


def test_reported_representative_is_midpoint_validated_with_its_own_generation_exposure():
    ctx = context()
    image = render(ctx)
    proof = support(image, ctx)
    output = fit.estimate_unit_face(image, ctx, proof, catalogue(), policy())
    assert output.status == 'supported'
    representative = output.representative
    assert representative.distance == pytest.approx(sum(output.distance_interval)/2)
    assert representative.phase == pytest.approx(tuple((start+width/2) % 1 for start, width in output.phase_intervals))
    assert output.distance_interval[1]-output.distance_interval[0] < .1*output.distance_interval[0]
    coords, _, _ = fit._rays(ctx, proof.generation_pixels)
    observed = image.reshape(-1, 3)[list(proof.generation_pixels)]/255
    face = next(f for f in catalogue().faces if f.sha256 == representative.face_sha256)
    texture, matrix, _ = fit._template(face, 2)
    error, gain, bias = fit._exposure(fit._sample(texture,
        (coords*representative.distance+representative.phase) @ matrix.T+.5), observed, policy())
    assert representative.generation_rmse == error
    assert representative.gain == gain and representative.bias == bias
    vcoords, _, _ = fit._rays(ctx, proof.validation_pixels)
    error, local = fit._validation(texture, matrix, vcoords,
        image.reshape(-1, 3)[list(proof.validation_pixels)]/255, representative, policy())
    assert representative.validation_rmse == error and representative.validation_max_tile_rmse == local
    assert output.residual_admission_threshold == pytest.approx(min(
        output.candidates[0].generation_rmse + policy().residual_slack + 1e-7,
        policy().generation_rmse_limit))
    assert len(output.cover) == output.retained_boxes > 0
    assert all(box.residual_lower_bound <= output.residual_admission_threshold for box in output.cover)
    assert all(box.geometry_sha256 == representative.geometry_sha256 for box in output.cover)
    # Known independently rendered truth is covered in the joint three-axis
    # union, not merely each marginal interval independently.
    assert any(box.distance[0] <= 3.23 <= box.distance[1]
               and box.phase_u[0] <= .173 <= box.phase_u[1]
               and box.phase_v[0] <= .287 <= box.phase_v[1] for box in output.cover)
    assert output.candidates[0].retained_geometry


def generated_catalogue(entries):
    """Independent generated archive: name, native pixels, UV turn, state count."""
    assets = {}
    for name, pixels, turn, count in entries:
        rgba = np.concatenate((pixels, np.full(pixels.shape[:2]+(1,), 255, np.uint8)), axis=2)
        stream = io.BytesIO()
        Image.fromarray(rgba).save(stream, format='PNG')
        assets[f'assets/minecraft/textures/block/{name}.png'] = stream.getvalue()
        assets[f'assets/minecraft/models/block/{name}.json'] = {
            'textures': {'all': f'block/{name}'}, 'elements': [{'from': [0, 0, 0], 'to': [16, 16, 16],
                'faces': {slot: {'texture': '#all', 'rotation': turn} for slot in SLOTS}}]}
        for state in range(count):
            assets[f'assets/minecraft/blockstates/{name}_{state:04}.json'] = {
                'variants': {'': {'model': f'block/{name}'}}}
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w') as zipped:
        for name, value in sorted(assets.items()):
            zipped.writestr(zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0)),
                            value if isinstance(value, bytes) else json.dumps(value))
    return load_unit_face_catalogue(archive.getvalue())


def face_identity(face):
    return face.block_id + '[' + ','.join(k+'='+v for k, v in face.properties) + ']|' + face.sha256


def assert_resource_refusal(output):
    assert output.status == 'abstained' and not output.search_complete
    assert output.representative is None and output.cover == ()
    assert output.distance_interval is None and output.phase_intervals is None
    assert output.retained_boxes == 0 and output.residual_admission_threshold is None
    assert all(c.retained_geometry is None for c in output.candidates)
    assert output.work_used <= output.work_budget
    assert output.actual_pixel_work <= output.pixel_work_budget


@pytest.fixture(scope='module')
def baseline_budget_case():
    ctx = context()
    image = render(ctx)
    proof = support(image, ctx)
    cat = catalogue()
    output = fit.estimate_unit_face(image, ctx, proof, cat, policy())
    assert output.status == 'supported'
    return ctx, image, proof, cat, output


def test_747_exact_state_aliases_preserve_every_identity_without_more_search(baseline_budget_case):
    ctx, image, proof, _, baseline = baseline_budget_case
    cat = catalogue(aliases=747)
    output = fit.estimate_unit_face(image, ctx, proof, cat, policy())
    assert output.status == 'supported'
    expected = tuple(sorted(face_identity(f) for f in cat.faces if f.face_slot == 'north'))
    assert len(expected) == 747
    assert output.candidates[0].state_aliases == output.representative.state_aliases == expected
    canonical_face = min((f for f in cat.faces if f.face_slot == 'north'), key=face_identity)
    assert output.candidates[0].face_sha256 == output.representative.face_sha256 == canonical_face.sha256
    assert output.compatible_faces == 747 and output.unique_templates == 1
    assert output.native_rgba_bytes == 64
    assert output.work_used == baseline.work_used
    assert output.actual_pixel_work == baseline.actual_pixel_work
    assert output.cover == baseline.cover


def test_over_64_unique_templates_matching_last_and_permutation_are_exhaustive(monkeypatch):
    entries = [(f'irrelevant_{i:03}', np.full((4, 4, 3), (i*3, 240-i*2, i), np.uint8), 0, 1)
               for i in range(65)] + [('zz_match', PATTERN, 0, 1)]
    cat = generated_catalogue(entries)
    assert [f.block_id for f in cat.faces if f.face_slot == 'north'][-1].startswith('minecraft:zz_match')
    ctx = context()
    image = render(ctx)
    proof = support(image, ctx)
    roots = set()
    native_calls = []
    original = fit._box_lower
    derivative = fit._derivative_bounds

    def record(texture, transform, coords, observed, box, fixed, derivatives=None):
        if np.array_equal(box, np.asarray((fixed.distance_bounds, (0, 1), (0, 1)))):
            roots.add((texture.tobytes(), tuple(transform.ravel())))
        return original(texture, transform, coords, observed, box, fixed, derivatives)

    def cache(texture):
        native_calls.append(texture.tobytes())
        return derivative(texture)

    monkeypatch.setattr(fit, '_box_lower', record)
    monkeypatch.setattr(fit, '_derivative_bounds', cache)
    output = fit.estimate_unit_face(image, ctx, proof, cat, policy())
    assert output.status == 'supported', output.reason
    assert len(roots) == output.unique_templates == len(output.candidates) == 66
    assert output.compatible_faces == 66 and len(native_calls) == len(set(native_calls)) == 66
    assert sum(c.retained_geometry for c in output.candidates) == 1
    assert output.representative.state_aliases[0].startswith('minecraft:zz_match')
    roots.clear()
    native_calls.clear()
    permuted = replace(cat, faces=tuple(reversed(cat.faces)))
    repeat = fit.estimate_unit_face(image, ctx, proof, permuted, policy())
    assert repeat.to_record() == output.to_record()
    assert len(roots) == 66 and len(native_calls) == 66


def test_mixed_alias_orientations_and_native_sizes_keep_exact_provenance(monkeypatch):
    entries = [('matching', PATTERN, 0, 3), ('turned', PATTERN, 90, 2),
               ('small', np.full((2, 2, 3), 220, np.uint8), 0, 4),
               ('large', np.full((8, 8, 3), 50, np.uint8), 0, 2)]
    cat = generated_catalogue(entries)
    ctx = context()
    image = render(ctx)
    calls = []
    original = fit._derivative_bounds
    monkeypatch.setattr(fit, '_derivative_bounds', lambda t: (calls.append(t.shape), original(t))[1])
    output = fit.estimate_unit_face(image, ctx, support(image, ctx), cat, policy())
    assert output.status == 'supported'
    expected = sorted(face_identity(f) for f in cat.faces if f.face_slot == 'north')
    assert sorted(identity for c in output.candidates for identity in c.state_aliases) == expected
    assert all(c.state_aliases == tuple(sorted(c.state_aliases)) for c in output.candidates)
    assert output.compatible_faces == 11 and output.unique_templates == 4
    assert len(calls) == 3 and sorted(shape[0] for shape in calls) == [2, 4, 8]
    assert output.native_rgba_bytes == 4*(2*2+4*4+8*8)


@pytest.mark.parametrize('cap,reason', [
    ({'max_catalogue_faces': 5}, 'catalogue_faces_cap'),
    ({'max_compatible_faces': 1}, 'compatible_faces_cap'),
    ({'max_unique_templates': 1}, 'unique_template_cap'),
    ({'max_native_rgba_bytes': 63}, 'native_rgba_bytes_cap'),
    ({'max_work': 2}, 'root_work_preflight'),
    ({'max_pixel_work': 1}, 'estimated_pixel_work_cap'),
    ({'max_pixels_per_role': 48}, 'source_work_cap'),
])
def test_all_lowered_preflight_caps_refuse_before_rgb_and_float_allocation(monkeypatch, cap, reason):
    ctx = context()
    image = render(ctx)
    proof = support(image, ctx)
    cat = catalogue(turns=(0, 90))
    observed = []
    allocated = []
    monkeypatch.setattr(fit, 'observe_role_pixels', lambda *a: observed.append(a))
    monkeypatch.setattr(fit, '_native_texture', lambda f: allocated.append(f))
    output = fit.estimate_unit_face(image, ctx, proof, cat, policy(**cap))
    assert_resource_refusal(output)
    assert output.reason == reason and output.work_used == output.actual_pixel_work == 0
    assert not observed and not allocated
    assert output.source_generation_sha256 == output.source_validation_sha256 == ''


def test_exact_evaluation_and_estimated_pixel_budget_boundaries(baseline_budget_case, monkeypatch):
    ctx, image, proof, cat, baseline = baseline_budget_case
    ng, nv = len(proof.generation_pixels), len(proof.validation_pixels)
    exact = policy(max_work=baseline.work_used)
    estimate = (exact.max_work+7)*ng + 2*nv
    visits = []
    original = fit._sample
    monkeypatch.setattr(fit, '_sample', lambda t, q: (visits.append(len(q)), original(t, q))[1])
    output = fit.estimate_unit_face(image, ctx, proof, cat, replace(exact, max_pixel_work=estimate))
    assert output.status == 'supported'
    assert output.work_used == exact.max_work
    assert output.actual_pixel_work == sum(visits) == output.estimated_pixel_work == estimate
    assert output.pixel_work_version == fit.PIXEL_WORK_VERSION
    assert output.search_policy == replace(exact, max_pixel_work=estimate)
    assert output.to_record()['search_policy']['max_pixel_work'] == estimate
    visits.clear()
    insufficient = fit.estimate_unit_face(image, ctx, proof, cat, replace(exact, max_pixel_work=estimate-1))
    assert_resource_refusal(insufficient)
    assert insufficient.reason == 'estimated_pixel_work_cap' and not visits
    insufficient = fit.estimate_unit_face(image, ctx, proof, cat, policy(max_work=baseline.work_used-1))
    assert_resource_refusal(insufficient)
    assert insufficient.reason == 'search_work_exhausted'
    assert insufficient.work_used == baseline.work_used-2


@pytest.mark.parametrize('stage,occurrence', [('representative', 1), ('spatial', 1),
    ('derivatives', 1), ('derivatives', 6), ('validation', 1), ('validation', 2)])
def test_post_search_sampling_guards_strip_every_certificate(monkeypatch, baseline_budget_case, stage, occurrence):
    ctx, image, proof, cat, _ = baseline_budget_case
    original = fit._WorkBudget.charge
    count = 0
    sampled = []
    sample = fit._sample

    def exhaust(budget, pixels, *, evaluation=False, stage='search'):
        nonlocal count
        if stage == target:
            count += 1
            if count == occurrence:
                raise fit._BudgetAbort('post_search_work_exhausted')
        return original(budget, pixels, evaluation=evaluation, stage=stage)

    target = stage
    monkeypatch.setattr(fit._WorkBudget, 'charge', exhaust)
    monkeypatch.setattr(fit, '_sample', lambda t, q: (sampled.append(len(q)), sample(t, q))[1])
    output = fit.estimate_unit_face(image, ctx, proof, cat, policy())
    assert_resource_refusal(output)
    assert output.reason == 'post_search_work_exhausted'
    assert output.domain_search_complete
    assert count == occurrence and output.actual_pixel_work == sum(sampled)


def test_root_initialization_abort_and_between_children_never_certify(monkeypatch):
    ctx = context()
    image = render(ctx)
    proof = support(image, ctx)
    cat = catalogue(turns=(0, 90, 180, 270))
    original = fit._WorkBudget.charge
    count = 0

    def third_root(budget, pixels, *, evaluation=False, stage='search'):
        nonlocal count
        count += 1
        if count == 3:
            raise fit._BudgetAbort('search_work_exhausted')
        return original(budget, pixels, evaluation=evaluation, stage=stage)

    monkeypatch.setattr(fit._WorkBudget, 'charge', third_root)
    output = fit.estimate_unit_face(image, ctx, proof, cat, policy())
    assert_resource_refusal(output)
    assert output.work_used == 2 and output.unique_templates == 4
    assert not output.domain_search_complete
    monkeypatch.setattr(fit._WorkBudget, 'charge', original)
    output = fit.estimate_unit_face(image, ctx, proof, catalogue(), policy(max_work=3))
    assert_resource_refusal(output)
    assert output.work_used == 2 and output.actual_pixel_work == 2*len(proof.generation_pixels)


@pytest.mark.parametrize('slot,axis,yaw,pitch', [
    ('north', 2, 0, 0), ('south', 2, 180, 0), ('west', 0, -90, 0),
    ('east', 0, 90, 0), ('down', 1, 0, -90), ('up', 1, 0, 90)])
@pytest.mark.parametrize('turn', [0, 90, 180, 270])
def test_independent_six_cardinal_normals_and_uv_turns_have_joint_truth_cover(slot, axis, yaw, pitch, turn):
    ctx = context(yaw=yaw, pitch=pitch, axis=axis)
    image = render(ctx, slot=slot, turn=turn)
    cat = catalogue(turns=(turn,))
    output = fit.estimate_unit_face(image, ctx, support(image, ctx), cat, policy())
    assert output.status == 'supported', (slot, turn, output.reason)
    assert output.representative.face_sha256 in {f.sha256 for f in cat.faces if f.face_slot == slot}
    assert abs(output.representative.distance/3.23-1) < .05
    assert all(circle_error(a, b) < .1 for a, b in zip(output.representative.phase, (.173, .287)))
    assert any(box.distance[0] <= 3.23 <= box.distance[1]
        and box.phase_u[0] <= .173 <= box.phase_u[1]
        and box.phase_v[0] <= .287 <= box.phase_v[1] for box in output.cover)


@pytest.mark.parametrize('turn', [90, 270])
def test_independent_composed_variant_and_uv_turn_joint_fit(turn):
    ctx = context(pitch=-90, axis=1)
    image = render(ctx, slot='down', variant=(90, 90), turn=turn)
    cat = catalogue(turns=(turn,), variant=(90, 90))
    output = fit.estimate_unit_face(image, ctx, support(image, ctx), cat, policy())
    assert output.status == 'supported', output.reason
    face = next(f for f in cat.faces if f.sha256 == output.representative.face_sha256)
    assert face.source_face == 'north' and face.face_slot == 'down'
    assert abs(output.representative.distance/3.23-1) < .05
    assert any(box.distance[0] <= 3.23 <= box.distance[1]
        and box.phase_u[0] <= .173 <= box.phase_u[1]
        and box.phase_v[0] <= .287 <= box.phase_v[1] for box in output.cover)


def test_resolvable_native_16_texture_positive_has_joint_truth_inside_cover():
    # Coherent asymmetric native 16x16 synthesis; every value is generated at
    # that native texel centre, with independent mixed Fourier modes. This is
    # not a resized catalogue asset or a reduced-resolution estimator template.
    y, x = np.meshgrid((np.arange(16)+.5)*2*np.pi/16,
                       (np.arange(16)+.5)*2*np.pi/16, indexing='ij')
    pattern = np.rint(np.stack((128+60*np.sin(x)+35*np.cos(y)+15*np.sin(x+2*y+.3),
        128+55*np.cos(x+.6)+40*np.sin(y+.8)+17*np.sin(2*x-y+.2),
        128+45*np.sin(x+y+.17)+45*np.cos(2*x+.9)+20*np.sin(2*y+.8)), axis=-1)).astype(np.uint8)
    assert len(np.unique(pattern.reshape(-1, 3), axis=0)) == 256
    assert np.any(pattern[:, -1] != pattern[:, 0]) and np.any(pattern[-1] != pattern[0])
    ctx = context(shape=(224, 256), focal=260)
    image = render(ctx, pattern)
    output = fit.estimate_unit_face(image, ctx, support(image, ctx), catalogue(pattern), policy())
    assert output.status == 'supported', (output.reason, output.work_used)
    assert output.native_rgba_bytes == 1024
    assert abs(output.representative.distance/3.23-1) < .05
    assert all(circle_error(a, b) < .1 for a, b in zip(output.representative.phase, (.173, .287)))
    assert any(box.distance[0] <= 3.23 <= box.distance[1]
        and box.phase_u[0] <= .173 <= box.phase_u[1]
        and box.phase_v[0] <= .287 <= box.phase_v[1] for box in output.cover)


def test_resolvable_high_gradient_random_native_16_cap_remains_fail_closed():
    pattern = np.random.default_rng(61534).integers(20, 236, (16, 16, 3), dtype=np.uint8)
    ctx = context(shape=(224, 256), focal=260)
    image = render(ctx, pattern)
    output = fit.estimate_unit_face(image, ctx, support(image, ctx), catalogue(pattern), policy())
    assert_resource_refusal(output)
    assert output.reason == 'search_work_exhausted' and output.work_used == 99_999
    assert not output.domain_search_complete


@pytest.mark.parametrize('slot,axis', [('north', 2), ('south', 2), ('west', 0), ('east', 0), ('up', 1), ('down', 1)])
def test_wrong_signed_plane_or_inward_corner_basis_is_unsupported(slot, axis):
    face = next(f for f in catalogue().faces if f.face_slot == slot)
    corners = tuple(tuple(-v if a == axis else v for a, v in enumerate(c)) for c in face.texture_corners)
    with pytest.raises(ValueError, match='unsupported_asset'):
        fit._template(replace(face, texture_corners=corners), axis)
    with pytest.raises(ValueError, match='unsupported_asset'):
        fit._template(replace(face, texture_corners=(face.texture_corners[0], face.texture_corners[3],
                                                    face.texture_corners[2], face.texture_corners[1])), axis)


@pytest.mark.parametrize('field', ['sample_map', 'sample_digest', 'witness_map', 'witness_depth',
                                  'witness_taps', 'witness_digest', 'tap_role'])
def test_forged_complete_source_proofs_refuse_before_rgb_or_work(monkeypatch, field):
    ctx = context()
    image = render(ctx)
    proof = deepcopy(support(image, ctx))
    sample = proof.samples[0]
    witness = proof.witnesses[0]
    if field == 'sample_map':
        object.__setattr__(sample, 'source_map', (sample.source_map[0]+.5, sample.source_map[1]))
    elif field == 'sample_digest':
        object.__setattr__(sample, 'witness_sha256', 'b'*64)
    elif field == 'witness_map':
        value = np.array(np.frombuffer(witness.map_x, np.float32))
        value[0] += .25
        object.__setattr__(witness, 'map_x', value.tobytes())
    elif field == 'witness_depth':
        value = bytearray(witness.depth_valid)
        value[sample.sample_index] = 1-value[sample.sample_index]
        object.__setattr__(witness, 'depth_valid', bytes(value))
    elif field == 'witness_taps':
        value = np.array(np.frombuffer(witness.tap_weight_numerators, np.uint16))
        value[0] += 1
        object.__setattr__(witness, 'tap_weight_numerators', value.tobytes())
    elif field == 'witness_digest':
        object.__setattr__(witness, 'witness_sha256', 'c'*64)
    else:
        value = bytearray(proof.partition.roles)
        index = sample.taps[0][0]
        value[index] = 3-value[index]
        object.__setattr__(proof.partition, 'roles', bytes(value))
    seen = []
    monkeypatch.setattr(fit, 'observe_role_pixels', lambda *a: seen.append(a))
    output = fit.estimate_unit_face(image, ctx, proof, catalogue(), policy())
    assert output.status == 'abstained' and output.reason == 'source_support_unavailable'
    assert output.work_used == output.actual_pixel_work == 0 and not seen
    assert not output.search_complete and not output.domain_search_complete

