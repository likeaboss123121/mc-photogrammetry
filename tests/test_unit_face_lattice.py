"""In-memory synthetic contract controls; no fixture, asset, fitting or file IO."""
from dataclasses import FrozenInstanceError, replace
import hashlib

import numpy as np
import pytest

from mc_photogrammetry.reconstruction import lattice
from mc_photogrammetry.reconstruction import unit_face_lattice as u


SHA = 'a' * 64


def face(**changes):
    rgba = bytes([30, 90, 170, 255] * 6)
    values = dict(asset_sha256=SHA, model_sha256='b' * 64,
                  texture_sha256=hashlib.sha256(rgba).hexdigest(), model_id='synthetic:cube',
                  variant_id='static-0', face_slot='north', texture_size=(2, 3), rgba=rgba,
                  uv_rect=(0, 0, 1, 1), uv_rotation_quarter_turns=0, cube_bounds=u.UNIT_CUBE,
                  full_cube=True, opaque=True, static_texture=True, variant_kind='deterministic',
                  uvlock=False, connected_texture=False, pack_supported=True)
    values.update(changes)
    return u.CanonicalUnitFace(**values)


def masks():
    shape = (4, 8)
    g, v = np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)
    g[:, :3], v[:, 5:] = True, True
    usable = np.ones(shape, dtype=bool)
    region = usable.copy()
    return g, v, usable, region


def partition():
    return u.SourceTapPartition.from_masks(*masks(), conditioning_sha256=SHA)


def witness(xs=(0.25, 5.25), ys=(0.5, 0.5), valid=None):
    x = np.asarray([xs], dtype=np.float32)
    y = np.asarray([ys], dtype=np.float32)
    valid = np.ones_like(x, dtype=bool) if valid is None else np.asarray([valid], dtype=bool)
    return lattice.rectification_witness_v1(x, y, valid, source_shape=(4, 8))


def groups():
    return (u.DependencyGroup('generation-rgb', 'generation', ((0, 0),)),
            u.DependencyGroup('validation-rgb', 'validation', ((0, 1),)))


def support():
    return u.partition_tap_support(partition(), (witness(),), groups())


def alias(f=None, **changes):
    f = f or face()
    values = dict(alias_id='unit', face_sha256=f.identity_sha256,
                  blocks_per_period=1.0, distance_interval=(9.9, 10.1),
                  phase_u=(0.95, 0.1), phase_v=(0.1, 0.1))
    values.update(changes)
    return u.UnitAlias(**values)


def outcome(**changes):
    f = face()
    values = dict(status='supported', period_measured=True, unit_scale_supported=True,
                  phase_supported=True, placement_supported=False, abstention='', aliases=(alias(f),),
                  catalogue=u.UnitFaceCatalogue((f,), True), support=support(), search_complete=True)
    values.update(changes)
    return u.UnitFaceLatticeResult(**values)


def assert_atomic(result, reason):
    assert result.status == 'refused' and result.reason == reason
    assert result.partition is None and result.witnesses == ()
    assert result.samples == result.groups == result.generation_pixels == result.validation_pixels == ()


def test_canonical_face_binds_full_uv_model_variant_and_original_resolution():
    f = face()
    assert f.unavailable_reasons == ()
    assert f.texture_size == (2, 3)  # no presumed 16 texels per unit face
    assert f.identity_sha256 == face().identity_sha256
    assert replace(f, uv_rotation_quarter_turns=1).identity_sha256 != f.identity_sha256
    assert replace(f, variant_id='static-1').identity_sha256 != f.identity_sha256
    assert replace(f, model_sha256='c' * 64).identity_sha256 != f.identity_sha256
    with pytest.raises(FrozenInstanceError):
        f.opaque = False


def test_face_snapshots_nested_uv_geometry_and_texture_payload():
    rgba = bytearray(face().rgba)
    uv = [0, 0, 1, 1]
    bounds = [list(v) for v in u.UNIT_CUBE]
    f = face(rgba=rgba, uv_rect=uv, cube_bounds=bounds)
    digest = f.identity_sha256
    rgba[0] = 255
    uv[2] = 0.5
    bounds[0][0] = -1
    assert f.identity_sha256 == digest
    assert f.unavailable_reasons == ()


@pytest.mark.parametrize('changes,reason', [
    ({'asset_sha256': None}, 'missing_asset_model_variant_identity'),
    ({'model_id': ''}, 'missing_asset_model_variant_identity'),
    ({'variant_id': ''}, 'missing_asset_model_variant_identity'),
    ({'uv_rect': None}, 'noncanonical_or_missing_uv'),
    ({'uv_rect': (0, 0, 0.5, 1)}, 'noncanonical_or_missing_uv'),
    ({'uv_rect': (1, 0, 0, 1)}, 'noncanonical_or_missing_uv'),
    ({'uv_rotation_quarter_turns': None}, 'noncanonical_or_missing_uv'),
    ({'full_cube': False}, 'not_an_explicit_full_unit_cube'),
    ({'cube_bounds': None}, 'not_an_explicit_full_unit_cube'),
    ({'opaque': None}, 'nonopaque_or_missing_opacity'),
    ({'static_texture': False}, 'animated_or_missing_static_semantics'),
    ({'variant_kind': 'random'}, 'random_or_missing_variant_semantics'),
    ({'variant_kind': None}, 'random_or_missing_variant_semantics'),
    ({'uvlock': True}, 'unsupported_or_missing_uvlock'),
    ({'connected_texture': True}, 'connected_or_missing_texture_semantics'),
    ({'pack_supported': False}, 'unsupported_or_missing_pack'),
])
def test_missing_or_unsupported_semantics_stay_explicit(changes, reason):
    f = face(**changes)
    assert reason in f.unavailable_reasons
    catalogue = u.UnitFaceCatalogue((f,), True)
    assert catalogue.faces == (f,)
    with pytest.raises(u.UnitFaceContractError, match='unit_anchor_unavailable'):
        outcome(catalogue=catalogue, aliases=(alias(f),))
    recorded = outcome(status='abstained', unit_scale_supported=False, phase_supported=False,
                       abstention='unit_anchor_unavailable', catalogue=catalogue, aliases=(alias(f),))
    assert recorded.unit_scale_supported is False


def test_nonopaque_pixels_cannot_be_certified_by_an_opaque_flag():
    data = bytearray(face().rgba)
    data[3] = 0
    f = face(rgba=data, texture_sha256=hashlib.sha256(data).hexdigest())
    assert 'nonopaque_or_missing_opacity' in f.unavailable_reasons


@pytest.mark.parametrize('changes,reason', [
    ({'texture_sha256': '0' * 64}, 'texture_digest_mismatch'),
    ({'model_sha256': 'bad'}, 'invalid_asset_digest'),
    ({'texture_size': (0, 16)}, 'invalid_texture_shape'),
    ({'texture_size': (129, 129)}, 'invalid_texture_shape'),
    ({'uv_rect': (0, 0, float('nan'), 1)}, 'invalid_uv'),
    ({'uv_rotation_quarter_turns': True}, 'invalid_uv_rotation'),
    ({'opaque': 1}, 'invalid_semantics_flag'),
    ({'cube_bounds': ((0, 0, 0), (float('inf'), 1, 1))}, 'invalid_cube_bounds'),
])
def test_malformed_catalogue_evidence_is_typed_invalid(changes, reason):
    with pytest.raises(u.UnitFaceContractError, match=reason):
        face(**changes)


def test_catalogue_preserves_distinct_template_variants_and_rejects_duplicates():
    f = face()
    variants = [f, replace(f, variant_id='static-1'), replace(f, uv_rotation_quarter_turns=1)]
    catalogue = u.UnitFaceCatalogue(variants, True)
    variants.clear()
    assert len(catalogue.faces) == 3
    with pytest.raises(u.UnitFaceContractError, match='duplicate_catalogue_identity'):
        u.UnitFaceCatalogue((f, f), True)
    with pytest.raises(u.UnitFaceContractError, match='catalogue_cap_exceeded'):
        u.UnitFaceCatalogue(tuple(replace(f, variant_id=str(i)) for i in range(65)), True)


def test_original_mask_partition_is_disjoint_bound_and_mutation_isolated():
    g, v, usable, region = masks()
    p = u.SourceTapPartition.from_masks(g, v, usable, region, conditioning_sha256=SHA)
    digest = p.identity_sha256
    g[:] = False
    v[:] = True
    usable[:] = False
    region[:] = False
    assert p.identity_sha256 == digest
    assert p.roles.count(1) == 12 and p.roles.count(2) == 12
    assert p.conditioning_sha256 == SHA
    with pytest.raises(FrozenInstanceError):
        p.roles = bytes(32)


@pytest.mark.parametrize('mode,reason', [
    ('overlap', 'overlapping_source_roles'), ('masked', 'excluded_partition_pixel'),
    ('region', 'excluded_partition_pixel'), ('missing', 'missing_partition_role'),
    ('dtype', 'invalid_partition_mask'), ('shape', 'partition_shape_mismatch'),
])
def test_invalid_source_role_partitions_refuse(mode, reason):
    g, v, usable, region = masks()
    if mode == 'overlap': v[0, 0] = True
    if mode == 'masked': usable[0, 0] = False
    if mode == 'region': region[0, 0] = False
    if mode == 'missing': v[:] = False
    if mode == 'dtype': g = g.astype(np.uint8)
    if mode == 'shape': v = v[:1]
    with pytest.raises(u.UnitFaceContractError, match=reason):
        u.SourceTapPartition.from_masks(g, v, usable, region, conditioning_sha256=SHA)


def test_exact_fractional_taps_and_integer_taps_keep_their_source_indices():
    result = support()
    assert result.status == 'complete'
    assert result.generation_pixels == (0, 1, 8, 9)
    assert result.validation_pixels == (5, 6, 13, 14)
    assert result.samples[0].taps == ((0, 384), (1, 128), (8, 384), (9, 128))
    integer = u.partition_tap_support(partition(), (witness((0, 5), (0, 0)),), groups())
    assert integer.status == 'complete'
    assert integer.samples[0].taps == ((0, 1024),)
    assert not set(result.generation_pixels) & set(result.validation_pixels)


def test_two_sampling_densities_are_checked_as_one_source_dependency_union():
    ws = (witness((0, 5), (0, 0)), witness((1, 6), (1, 1)))
    gs = (u.DependencyGroup('g-density-union', 'generation', ((0, 0), (1, 0))),
          u.DependencyGroup('v-density-union', 'validation', ((0, 1), (1, 1))))
    result = u.partition_tap_support(partition(), ws, gs)
    assert result.status == 'complete'
    assert result.generation_pixels == (0, 9)
    assert result.validation_pixels == (5, 14)
    # Distinct rectified samples can reuse a source tap across role/density.
    ws = (witness((0, 5), (0, 0)), witness((1, 0), (1, 0)))
    assert_atomic(u.partition_tap_support(partition(), ws, gs), 'overlapping_source_taps')


def test_overlapping_fractional_taps_are_refused_without_publishing_a_prefix():
    result = u.partition_tap_support(partition(), (witness((0.25, 0.75)),), groups())
    assert_atomic(result, 'overlapping_source_taps')


def test_gradient_stencil_checks_both_endpoints_and_not_just_first_weight():
    gs = (u.DependencyGroup('generation-gradient', 'generation', ((0, 0), (0, 1))),
          u.DependencyGroup('validation', 'validation', ((0, 2),)))
    w = witness((0, 5, 7), (0, 0, 0))
    assert_atomic(u.partition_tap_support(partition(), (w,), gs), 'cross_partition_stencil')
    valid = witness((0, 1, 7), (0, 0, 0))
    result = u.partition_tap_support(partition(), (valid,), gs)
    assert result.status == 'complete' and result.generation_pixels == (0, 1)


@pytest.mark.parametrize('xs,ys,valid,reason', [
    ((-0.25, 5), (0, 0), None, 'border_source_tap'),
    ((2.5, 5), (0, 0), None, 'cross_partition_stencil'),
    ((0, 5), (0, 0), (False, True), 'invalid_depth_sample'),
])
def test_border_excluded_and_invalid_depth_taps_refuse(xs, ys, valid, reason):
    assert_atomic(u.partition_tap_support(partition(), (witness(xs, ys, valid),), groups()), reason)


def test_gui_and_region_pixels_are_excluded_even_when_a_warp_covers_them():
    for kind in ('gui', 'region'):
        g, v, usable, region = masks()
        g[0, 0] = False
        if kind == 'gui': usable[0, 0] = False
        else: region[0, 0] = False
        p = u.SourceTapPartition.from_masks(g, v, usable, region, conditioning_sha256=SHA)
        assert_atomic(u.partition_tap_support(p, (witness((0, 5), (0, 0)),), groups()),
                      'excluded_source_tap')


def test_missing_truncated_and_late_invalid_witnesses_refuse_atomically():
    missing = lattice.WitnessRefusalV1('refused', 'budget_exhausted', 'rectified_samples', 0, 2, 0)
    assert_atomic(u.partition_tap_support(partition(), (missing,), groups()), 'witness_unavailable')
    w = witness()
    gs = groups() + (u.DependencyGroup('late-v', 'validation', ((1, 1),)),)
    bad = witness()
    bad.tap_weight_numerators.flags.writeable = True
    assert_atomic(u.partition_tap_support(partition(), (w, bad), gs), 'mutable_witness')
    absent = (u.DependencyGroup('g', 'generation', ((0, 0),)),
              u.DependencyGroup('v', 'validation', ((0, 9),)))
    assert_atomic(u.partition_tap_support(partition(), (w,), absent), 'sample_reference_unavailable')


def test_a_self_consistent_but_false_tap_table_is_replayed_and_rejected():
    w = witness()
    indices = w.tap_source_indices.copy()
    indices[0] = 2
    indices.flags.writeable = False
    counterfeit = replace(w, tap_source_indices=indices, witness_sha256='')
    assert counterfeit.witness_sha256 != w.witness_sha256
    assert_atomic(u.partition_tap_support(partition(), (counterfeit,), groups()), 'witness_replay_mismatch')


def test_support_snapshots_do_not_alias_the_witness_maps_or_taps():
    w = witness()
    result = u.partition_tap_support(partition(), (w,), groups())
    digest, saved = result.identity_sha256, result.samples[0].taps
    w.map_x.flags.writeable = True
    w.map_x[:] = 100
    w.tap_source_indices.flags.writeable = True
    w.tap_source_indices[:] = -1
    assert result.identity_sha256 == digest
    assert result.samples[0].taps == saved


@pytest.mark.parametrize('caps,reason', [
    ({'max_source_pixels': 31}, 'source_pixel_cap_exceeded'),
    ({'max_rectified_samples': 1}, 'sample_cap_exceeded'),
    ({'max_bilinear_taps': 7}, 'tap_cap_exceeded'),
    ({'max_dependency_memberships': 1}, 'dependency_cap_exceeded'),
])
def test_caps_are_preflighted_before_witness_replay(monkeypatch, caps, reason):
    w = witness()
    def forbidden(*_a, **_k):
        pytest.fail('cap refusal must precede witness replay')
    monkeypatch.setattr(lattice, 'rectification_witness_v1', forbidden)
    assert_atomic(u.partition_tap_support(partition(), (w,), groups(),
                  limits=replace(u.LIMITS, **caps)), reason)


def test_exact_work_caps_accept_and_invalid_limit_settings_refuse():
    caps = replace(u.LIMITS, max_source_pixels=32, max_rectified_samples=2,
                   max_bilinear_taps=8, max_dependency_memberships=2)
    assert u.partition_tap_support(partition(), (witness(),), groups(), limits=caps).status == 'complete'
    with pytest.raises(u.UnitFaceContractError, match='invalid_contract_limits'):
        replace(u.LIMITS, max_rectified_samples=4097)


def test_role_pixel_observations_ignore_other_role_and_masked_mutations():
    s = support()
    image = np.arange(4 * 8 * 3, dtype=np.uint8).reshape(4, 8, 3)
    g = u.observe_role_pixels(image, s, 'generation')
    v = u.observe_role_pixels(image, s, 'validation')
    image.reshape(-1, 3)[list(s.validation_pixels)] = 255
    image.reshape(-1, 3)[[3, 4, 31]] = 254
    assert u.observe_role_pixels(image, s, 'generation') == g
    assert u.observe_role_pixels(image, s, 'validation').identity_sha256 != v.identity_sha256
    assert g.rgb == bytes(range(6)) + bytes(range(24, 30))
    saved = g.rgb
    image[:] = 0
    assert g.rgb == saved


def test_observation_cannot_read_without_complete_source_support():
    with pytest.raises(u.UnitFaceContractError, match='support_unavailable'):
        u.observe_role_pixels(np.ones((4, 8, 3), dtype=np.uint8),
                              u.PartitionedSupport('refused', 'witness_unavailable'), 'generation')
    with pytest.raises(u.UnitFaceContractError, match='invalid_source_image'):
        u.observe_role_pixels(np.ones((4, 8, 3), dtype=float), support(), 'generation')


def test_supported_declaration_separates_flags_and_never_authorizes_geometry():
    result = outcome()
    assert result.period_measured and result.unit_scale_supported and result.phase_supported
    assert result.placement_supported is False
    assert result.camera_scale_known is False and result.reconstruction_authorized is False
    assert result.validation_scope == 'conditional_on_frozen_upstream_context'
    assert result.identity_sha256 == outcome().identity_sha256
    assert outcome(phase_supported=False, aliases=(alias(phase_u=None, phase_v=None),)).phase_supported is False
    with pytest.raises(FrozenInstanceError): result.reconstruction_authorized = True


def test_period_and_two_unit_aliases_do_not_manufacture_unique_scale_or_phase():
    a = alias()
    rival = replace(a, alias_id='texel', blocks_per_period=1/16,
                    distance_interval=(0.61, 0.64), phase_u=None, phase_v=None)
    result = outcome(status='ambiguous', unit_scale_supported=False, phase_supported=False,
                     abstention='ambiguous_unit_alias', aliases=(a, rival))
    assert result.aliases == (a, rival) and result.period_measured
    with pytest.raises(u.UnitFaceContractError, match='invalid_supported_result'):
        outcome(aliases=(a, rival))


def test_missing_template_remains_typed_abstention_even_with_measured_period():
    result = outcome(status='abstained', unit_scale_supported=False, phase_supported=False,
                     abstention='unit_anchor_unavailable', aliases=(), catalogue=u.UnitFaceCatalogue((), True))
    assert result.period_measured and not result.unit_scale_supported and not result.phase_supported
    absent_period = replace(result, period_measured=False, abstention='period_unavailable')
    assert absent_period.aliases == ()


@pytest.mark.parametrize('changes,reason', [
    ({'period_measured': False}, 'scale_without_period'),
    ({'unit_scale_supported': False}, 'phase_without_unit_scale'),
    ({'placement_supported': True}, 'placement_adapter_unavailable'),
    ({'period_measured': 1}, 'invalid_result_flag'),
    ({'abstention': 'wrong'}, 'invalid_supported_result'),
    ({'aliases': ()}, 'invalid_supported_result'),
    ({'support': u.PartitionedSupport('refused', 'bad')}, 'invalid_supported_result'),
    ({'aliases': (alias(face_sha256='0' * 64),)}, 'unit_anchor_unavailable'),
    ({'aliases': (alias(phase_u=(0, 1)),)}, 'phase_evidence_unavailable'),
])
def test_invalid_supported_result_combinations_are_refused(changes, reason):
    with pytest.raises(u.UnitFaceContractError, match=reason): outcome(**changes)


@pytest.mark.parametrize('which', ['search', 'catalogue'])
def test_incomplete_search_or_catalogue_cannot_publish_unique_or_partial_aliases(which):
    changes = {'search_complete': False} if which == 'search' else {'catalogue': u.UnitFaceCatalogue((face(),), False)}
    with pytest.raises(u.UnitFaceContractError, match='truncated_result_not_atomic'):
        outcome(**changes)
    result = outcome(status='abstained', unit_scale_supported=False, phase_supported=False,
                     abstention='search_incomplete', aliases=(), **changes)
    assert result.period_measured and result.aliases == ()


@pytest.mark.parametrize('changes,reason', [
    ({'blocks_per_period': float('inf')}, 'invalid_alias_scale'),
    ({'blocks_per_period': True}, 'invalid_alias_scale'),
    ({'distance_interval': (0, 1)}, 'invalid_distance_interval'),
    ({'distance_interval': (2, 1)}, 'invalid_distance_interval'),
    ({'phase_u': (float('nan'), 0.1)}, 'invalid_phase_interval'),
    ({'phase_v': (1, 0.1)}, 'invalid_phase_interval'),
    ({'phase_v': (0, -0.1)}, 'invalid_phase_interval'),
])
def test_invalid_alias_intervals_and_nonfinite_values_refuse(changes, reason):
    with pytest.raises(u.UnitFaceContractError, match=reason): alias(**changes)


def test_result_snapshots_alias_lists_and_retains_circular_wrap_intervals():
    aliases = [alias()]
    result = outcome(aliases=aliases)
    aliases.clear()
    assert result.aliases[0].phase_u == (0.95, 0.1)  # wraps zero, not a snapped seam
    with pytest.raises(u.UnitFaceContractError, match='duplicate_alias_identity'):
        outcome(aliases=(alias(), alias()))


def test_refusal_payloads_and_direct_support_records_cannot_bypass_atomicity():
    with pytest.raises(u.UnitFaceContractError, match='non_atomic_support_refusal'):
        u.PartitionedSupport('refused', 'bad', partition())
    s = support()
    with pytest.raises(u.UnitFaceContractError, match='cross_partition_stencil'):
        u.PartitionedSupport('complete', '', s.partition, s.samples,
            (u.DependencyGroup('g', 'generation', ((0, 1),)),
             u.DependencyGroup('v', 'validation', ((0, 0),))), s.witnesses)


def test_direct_support_construction_replays_source_maps_not_just_disjoint_claims():
    s = support()
    falsified = replace(s.samples[0], taps=((0, 1024),))
    with pytest.raises(u.UnitFaceContractError, match='sample_tap_replay_mismatch'):
        u.PartitionedSupport('complete', '', s.partition, (falsified, s.samples[1]), s.groups, s.witnesses)


def test_invalid_support_cannot_preserve_ambiguous_or_abstained_alias_payloads():
    a = alias()
    refused = u.PartitionedSupport('refused', 'bad')
    with pytest.raises(u.UnitFaceContractError, match='ambiguity_without_source_support'):
        outcome(status='ambiguous', unit_scale_supported=False, phase_supported=False,
                abstention='ambiguous_unit_alias', aliases=(a, replace(a, alias_id='rival')),
                support=refused)
    with pytest.raises(u.UnitFaceContractError, match='invalid_support_abstention'):
        outcome(status='abstained', unit_scale_supported=False, phase_supported=False,
                abstention='support_unavailable', support=refused)
    result = outcome(status='abstained', unit_scale_supported=False, phase_supported=False,
                     abstention='support_unavailable', support=refused, aliases=())
    assert result.aliases == () and result.unit_scale_supported is False


def test_ambiguous_aliases_must_reference_retained_catalogue_templates():
    a = alias()
    with pytest.raises(u.UnitFaceContractError, match='unknown_alias_template'):
        outcome(status='ambiguous', unit_scale_supported=False, phase_supported=False,
                abstention='ambiguous_unit_alias', aliases=(a, replace(a, alias_id='rival',
                                                                 face_sha256='0' * 64)))


@pytest.mark.parametrize('all_hashes', [False, True])
def test_forged_original_witness_hash_cannot_enter_direct_complete_or_supported_result(all_hashes):
    s = support()
    samples = tuple(replace(sample, witness_sha256='0' * 64)
                    if all_hashes or index == 0 else sample for index, sample in enumerate(s.samples))
    # Exact Astra repro: the old direct constructor has no original proof.
    with pytest.raises(u.UnitFaceContractError, match='witness_proof_unavailable'):
        forged = u.PartitionedSupport('complete', '', s.partition, samples, s.groups)
        outcome(support=forged)
    # Retaining the authentic proof still rejects a changed digest, whether one
    # same-index sample conflicts with its peer or both claim a consistent lie.
    with pytest.raises(u.UnitFaceContractError, match='sample_witness_binding_mismatch'):
        forged = u.PartitionedSupport('complete', '', s.partition, samples, s.groups, s.witnesses)
        outcome(support=forged)
    with pytest.raises(u.UnitFaceContractError, match='sample_witness_binding_mismatch'):
        replace(s, samples=samples)


@pytest.mark.parametrize('index', [1, 99])
def test_changed_sample_index_and_matching_group_reference_still_join_original_witness(index):
    # A third valid pixel prevents rejection just for a duplicate sample key.
    w = witness((0.25, 1.25, 5.25), (0.5, 0.5, 0.5))
    gs = (u.DependencyGroup('g', 'generation', ((0, 0),)),
          u.DependencyGroup('v', 'validation', ((0, 2),)))
    s = u.partition_tap_support(partition(), (w,), gs)
    assert s.status == 'complete'
    samples = (replace(s.samples[0], sample_index=index), s.samples[1])
    altered = (replace(s.groups[0], sample_refs=((0, index),)), s.groups[1])
    reason = 'sample_map_binding_mismatch' if index == 1 else 'sample_index_binding_mismatch'
    with pytest.raises(u.UnitFaceContractError, match=reason):
        forged = u.PartitionedSupport('complete', '', s.partition, samples, altered, s.witnesses)
        outcome(support=forged)


def test_original_witness_snapshot_is_replayable_immutable_and_complete():
    w = witness((0.25, 1.25, 5.25), (0.5, 0.5, 0.5))
    gs = (u.DependencyGroup('g', 'generation', ((0, 0),)),
          u.DependencyGroup('v', 'validation', ((0, 2),)))
    s = u.partition_tap_support(partition(), (w,), gs)
    assert s.status == 'complete'
    proof = s.witnesses[0]
    assert proof.witness_sha256 == w.witness_sha256 and proof.map_shape == (1, 3)
    assert proof.sample_payload(1)[0] == (1.25, 0.5)  # unselected original sample retained
    assert proof.sample_payload(0) == (s.samples[0].source_map, s.samples[0].depth_valid, s.samples[0].taps)
    assert u.PartitionedSupport('complete', '', s.partition, s.samples, s.groups, s.witnesses) == s
    assert outcome(support=s).reconstruction_authorized is False
    before = s.identity_sha256
    w.map_x.flags.writeable = True
    w.map_x[:] = 100
    w.tap_source_indices.flags.writeable = True
    w.tap_source_indices[:] = -1
    assert s.identity_sha256 == before and proof.sample_payload(1)[0] == (1.25, 0.5)
    with pytest.raises(FrozenInstanceError): proof.witness_sha256 = '0' * 64
    with pytest.raises(u.UnitFaceContractError, match='witness_proof_replay_mismatch'):
        replace(proof, witness_sha256='0' * 64)
    changed = bytearray(proof.map_x)
    changed[:4] = np.asarray([2.25], dtype=np.float32).tobytes()
    with pytest.raises(u.UnitFaceContractError, match='witness_proof_replay_mismatch'):
        replace(proof, map_x=bytes(changed))


def test_proof_payload_cap_precedes_any_replay_and_accepts_exact_boundary(monkeypatch):
    p, w, gs = partition(), witness(), groups()
    complete = u.partition_tap_support(p, (w,), gs)
    payload_bytes = complete.witnesses[0].payload_bytes
    caps = replace(u.LIMITS, max_witness_payload_bytes=payload_bytes)
    assert u.partition_tap_support(p, (w,), gs, limits=caps).status == 'complete'
    def forbidden(*_a, **_k):
        pytest.fail('proof byte cap must be checked before replay')
    monkeypatch.setattr(lattice, 'rectification_witness_v1', forbidden)
    assert_atomic(u.partition_tap_support(p, (w,), gs,
                  limits=replace(caps, max_witness_payload_bytes=payload_bytes-1)),
                  'witness_payload_cap_exceeded')
