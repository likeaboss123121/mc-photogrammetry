"""Private synthetic unit-face contracts; no estimator, assets or production caller.

Catalogue semantics are declarations bound to bytes, not authenticated asset
provenance. Support independence is conditional on frozen upstream context and
on the caller declaring every operation's complete stencil dependencies. This
module neither derives that context nor fits periods, templates or photometry.
A supported result is a provisional, structurally checked evidence declaration;
it is not a pixel-match certificate, recovered camera scale or placement permit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math

import numpy as np

from . import lattice


class UnitFaceContractError(ValueError):
    """Typed malformed/unsupported contract input, with a stable reason."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def _require(condition, reason):
    if not condition:
        raise UnitFaceContractError(reason)


def _hex(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _digest(value):
    def encode(item):
        if isinstance(item, bytes):
            return {'bytes_sha256': hashlib.sha256(item).hexdigest(), 'length': len(item)}
        raise TypeError('unsupported contract digest value')
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False, default=encode).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class ContractLimits:
    """Lowerable synthetic-work caps, not a peak/RSS allocation guarantee."""

    max_source_pixels: int = 65_536
    max_rectified_samples: int = 4_096
    max_bilinear_taps: int = 16_384
    max_dependency_memberships: int = 8_192
    max_templates: int = 64
    max_texture_side: int = 128
    max_aliases: int = 32
    max_witness_payload_bytes: int = 1_048_576

    def __post_init__(self):
        ceilings = (65_536, 4_096, 16_384, 8_192, 64, 128, 32, 1_048_576)
        for value, ceiling in zip(asdict(self).values(), ceilings):
            _require(type(value) is int and 0 <= value <= ceiling, 'invalid_contract_limits')


LIMITS = ContractLimits()
FACE_SLOTS = ('down', 'up', 'north', 'south', 'west', 'east')
UNIT_CUBE = ((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))


@dataclass(frozen=True, slots=True)
class CanonicalUnitFace:
    """One explicitly model/UV/variant-bound full-face declaration.

    RGBA keeps original resolution; no 16-texel size assumption or thumbnail
    conversion exists. Missing or unsupported semantics remain in the catalogue
    and make this record unavailable as a unit anchor.
    """

    asset_sha256: str | None
    model_sha256: str | None
    texture_sha256: str | None
    model_id: str
    variant_id: str
    face_slot: str
    texture_size: tuple[int, int]  # height, width
    rgba: bytes
    uv_rect: tuple[float, float, float, float] | None
    uv_rotation_quarter_turns: int | None
    cube_bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None
    full_cube: bool | None
    opaque: bool | None
    static_texture: bool | None
    variant_kind: str | None
    uvlock: bool | None
    connected_texture: bool | None
    pack_supported: bool | None

    def __post_init__(self):
        for value in (self.asset_sha256, self.model_sha256, self.texture_sha256):
            _require(value is None or _hex(value), 'invalid_asset_digest')
        _require(type(self.model_id) is str and type(self.variant_id) is str
                 and len(self.model_id) <= 256 and len(self.variant_id) <= 256,
                 'invalid_model_identity')
        _require(type(self.face_slot) is str and self.face_slot in FACE_SLOTS, 'invalid_face_slot')
        _require(type(self.texture_size) in (tuple, list) and len(self.texture_size) == 2,
                 'invalid_texture_shape')
        shape = tuple(self.texture_size)
        _require(all(type(n) is int and 0 < n <= LIMITS.max_texture_side for n in shape),
                 'invalid_texture_shape')
        _require(type(self.rgba) in (bytes, bytearray) and len(self.rgba) == shape[0] * shape[1] * 4,
                 'invalid_texture_payload')
        payload = bytes(self.rgba)
        if self.texture_sha256 is not None:
            _require(hashlib.sha256(payload).hexdigest() == self.texture_sha256,
                     'texture_digest_mismatch')
        object.__setattr__(self, 'texture_size', shape)
        object.__setattr__(self, 'rgba', payload)
        if self.uv_rect is not None:
            _require(type(self.uv_rect) in (tuple, list) and len(self.uv_rect) == 4
                     and all(_finite(v) for v in self.uv_rect), 'invalid_uv')
            object.__setattr__(self, 'uv_rect', tuple(self.uv_rect))
        _require(self.uv_rotation_quarter_turns is None or
                 type(self.uv_rotation_quarter_turns) is int and
                 0 <= self.uv_rotation_quarter_turns < 4, 'invalid_uv_rotation')
        if self.cube_bounds is not None:
            _require(type(self.cube_bounds) in (tuple, list) and len(self.cube_bounds) == 2
                     and all(type(v) in (tuple, list) and len(v) == 3
                             and all(_finite(x) for x in v) for v in self.cube_bounds),
                     'invalid_cube_bounds')
            object.__setattr__(self, 'cube_bounds', tuple(tuple(v) for v in self.cube_bounds))
        for name in ('full_cube', 'opaque', 'static_texture', 'uvlock',
                     'connected_texture', 'pack_supported'):
            _require(getattr(self, name) is None or type(getattr(self, name)) is bool,
                     'invalid_semantics_flag')
        _require(self.variant_kind in (None, 'deterministic', 'random'), 'invalid_variant_kind')

    @property
    def unavailable_reasons(self) -> tuple[str, ...]:
        reasons = []
        if (None in (self.asset_sha256, self.model_sha256, self.texture_sha256)
                or not self.model_id or not self.variant_id):
            reasons.append('missing_asset_model_variant_identity')
        if self.uv_rect != (0, 0, 1, 1) or self.uv_rotation_quarter_turns is None:
            reasons.append('noncanonical_or_missing_uv')
        if self.cube_bounds != UNIT_CUBE or self.full_cube is not True:
            reasons.append('not_an_explicit_full_unit_cube')
        if self.opaque is not True or any(v != 255 for v in self.rgba[3::4]):
            reasons.append('nonopaque_or_missing_opacity')
        if self.static_texture is not True:
            reasons.append('animated_or_missing_static_semantics')
        if self.variant_kind != 'deterministic':
            reasons.append('random_or_missing_variant_semantics')
        if self.uvlock is not False:
            reasons.append('unsupported_or_missing_uvlock')
        if self.connected_texture is not False:
            reasons.append('connected_or_missing_texture_semantics')
        if self.pack_supported is not True:
            reasons.append('unsupported_or_missing_pack')
        return tuple(reasons)

    @property
    def identity_sha256(self):
        return _digest(asdict(self))


@dataclass(frozen=True, slots=True)
class UnitFaceCatalogue:
    faces: tuple[CanonicalUnitFace, ...]
    complete: bool

    def __post_init__(self):
        _require(type(self.faces) in (tuple, list) and len(self.faces) <= LIMITS.max_templates,
                 'catalogue_cap_exceeded')
        faces = tuple(self.faces)
        _require(all(type(face) is CanonicalUnitFace for face in faces), 'invalid_catalogue_face')
        _require(type(self.complete) is bool, 'invalid_catalogue_completeness')
        identities = tuple(face.identity_sha256 for face in faces)
        _require(len(set(identities)) == len(identities), 'duplicate_catalogue_identity')
        object.__setattr__(self, 'faces', faces)

    @property
    def identity_sha256(self):
        return _digest({'faces': [f.identity_sha256 for f in self.faces], 'complete': self.complete})


@dataclass(frozen=True, slots=True)
class SourceTapPartition:
    """Immutable original-pixel roles; 0 excluded, 1 generation, 2 validation."""

    source_shape: tuple[int, int]
    usable: bytes
    region: bytes
    roles: bytes
    conditioning_sha256: str

    def __post_init__(self):
        _require(type(self.source_shape) in (tuple, list) and len(self.source_shape) == 2
                 and all(type(n) is int and n > 0 for n in self.source_shape), 'invalid_source_shape')
        shape = tuple(self.source_shape)
        count = shape[0] * shape[1]
        _require(count <= LIMITS.max_source_pixels, 'source_pixel_cap_exceeded')
        _require(_hex(self.conditioning_sha256), 'invalid_conditioning_digest')
        for name in ('usable', 'region', 'roles'):
            value = getattr(self, name)
            _require(type(value) in (bytes, bytearray) and len(value) == count, 'invalid_partition_payload')
            object.__setattr__(self, name, bytes(value))
        _require(all(v in (0, 1) for v in self.usable + self.region)
                 and all(v in (0, 1, 2) for v in self.roles), 'invalid_partition_role')
        _require(all(not role or self.usable[i] and self.region[i]
                     for i, role in enumerate(self.roles)), 'excluded_partition_pixel')
        _require(1 in self.roles and 2 in self.roles, 'missing_partition_role')
        object.__setattr__(self, 'source_shape', shape)

    @classmethod
    def from_masks(cls, generation, validation, usable, region, *, conditioning_sha256):
        masks = (generation, validation, usable, region)
        _require(all(isinstance(m, np.ndarray) and m.dtype == np.bool_ and m.ndim == 2
                     for m in masks), 'invalid_partition_mask')
        _require(all(m.shape == generation.shape for m in masks), 'partition_shape_mismatch')
        _require(generation.size <= LIMITS.max_source_pixels, 'source_pixel_cap_exceeded')
        _require(not np.any(generation & validation), 'overlapping_source_roles')
        roles = generation.astype(np.uint8) + 2 * validation.astype(np.uint8)
        return cls(generation.shape, usable.astype(np.uint8).tobytes(),
                   region.astype(np.uint8).tobytes(), roles.tobytes(), conditioning_sha256)

    @property
    def identity_sha256(self):
        return _digest(asdict(self))


@dataclass(frozen=True, slots=True)
class DependencyGroup:
    """All rectified-sample dependencies of one operation, including its stencil.

    An edge/gradient group must include both endpoints. This schema checks the
    supplied dependency closure; it cannot discover an omitted caller stencil.
    """

    operation_id: str
    role: str
    sample_refs: tuple[tuple[int, int], ...]  # witness index, flattened sample

    def __post_init__(self):
        _require(type(self.operation_id) is str and 0 < len(self.operation_id) <= 256,
                 'invalid_operation_identity')
        _require(type(self.role) is str and self.role in ('generation', 'validation'), 'invalid_dependency_role')
        _require(type(self.sample_refs) in (tuple, list) and
                 0 < len(self.sample_refs) <= LIMITS.max_dependency_memberships,
                 'invalid_dependency_memberships')
        refs = []
        for pair in self.sample_refs:
            _require(type(pair) in (tuple, list) and len(pair) == 2
                     and all(type(i) is int and i >= 0 for i in pair), 'invalid_sample_reference')
            refs.append(tuple(pair))
        _require(len(set(refs)) == len(refs), 'duplicate_sample_reference')
        object.__setattr__(self, 'sample_refs', tuple(refs))


@dataclass(frozen=True, slots=True)
class TapSample:
    witness_index: int
    sample_index: int
    witness_sha256: str
    source_map: tuple[float, float]
    depth_valid: bool
    taps: tuple[tuple[int, int], ...]  # nonzero source index, numerator / 1024

    def __post_init__(self):
        _require(type(self.witness_index) is int and self.witness_index >= 0
                 and type(self.sample_index) is int and self.sample_index >= 0
                 and _hex(self.witness_sha256), 'invalid_tap_identity')
        _require(type(self.source_map) is tuple and len(self.source_map) == 2
                 and all(_finite(v) for v in self.source_map), 'invalid_tap_map')
        _require(type(self.depth_valid) is bool and type(self.taps) is tuple
                 and 0 < len(self.taps) <= 4, 'invalid_tap_payload')
        _require(all(type(t) is tuple and len(t) == 2 and type(t[0]) is int
                     and t[0] >= -1 and type(t[1]) is int and 0 < t[1] <= 1024
                     for t in self.taps) and sum(t[1] for t in self.taps) == 1024,
                 'invalid_tap_weights')


@dataclass(frozen=True, slots=True)
class WitnessProof:
    """Replayable immutable snapshot of one COMPLETE original tap witness.

    Map shape/order and original metadata digest remain bound, including samples
    not selected by an operation. This binds supplied witnesses, not their origin.
    The byte cap counts retained array/record payloads, not Python objects/RSS.
    """

    source_shape: tuple[int, int]
    map_shape: tuple[int, int]
    map_x: bytes
    map_y: bytes
    depth_valid: bytes
    tap_ptr: bytes
    tap_source_indices: bytes
    tap_weight_numerators: bytes
    record_json: str
    witness_sha256: str

    def __post_init__(self):
        for shape in (self.source_shape, self.map_shape):
            _require(type(shape) is tuple and len(shape) == 2
                     and all(type(n) is int and n > 0 for n in shape), 'invalid_witness_proof_shape')
        n = self.map_shape[0] * self.map_shape[1]
        _require(n <= LIMITS.max_rectified_samples and
                 self.source_shape[0] * self.source_shape[1] <= LIMITS.max_source_pixels,
                 'witness_proof_cap_exceeded')
        payloads = (self.map_x, self.map_y, self.depth_valid, self.tap_ptr,
                    self.tap_source_indices, self.tap_weight_numerators)
        _require(all(type(v) is bytes for v in payloads) and type(self.record_json) is str
                 and _hex(self.witness_sha256), 'invalid_witness_proof_payload')
        _require(self.payload_bytes <= LIMITS.max_witness_payload_bytes,
                 'witness_payload_cap_exceeded')
        _require(len(self.map_x) == len(self.map_y) == n * 4 and len(self.depth_valid) == n
                 and all(v in (0, 1) for v in self.depth_valid)
                 and len(self.tap_ptr) == (n + 1) * 8
                 and len(self.tap_source_indices) % 8 == 0
                 and len(self.tap_weight_numerators) * 4 == len(self.tap_source_indices)
                 and len(self.tap_source_indices) <= LIMITS.max_bilinear_taps * 8,
                 'invalid_witness_proof_payload')
        xs = np.frombuffer(self.map_x, dtype=np.float32).reshape(self.map_shape)
        ys = np.frombuffer(self.map_y, dtype=np.float32).reshape(self.map_shape)
        depth = np.frombuffer(self.depth_valid, dtype=np.bool_).reshape(self.map_shape)
        _require(np.isfinite(xs).all() and np.isfinite(ys).all(), 'invalid_witness_proof_maps')
        replay = lattice.rectification_witness_v1(xs, ys, depth, source_shape=self.source_shape,
            max_samples=LIMITS.max_rectified_samples, max_taps=LIMITS.max_bilinear_taps)
        _require(replay.witness_sha256 == self.witness_sha256
                 and _witness_record_json(replay) == self.record_json
                 and replay.tap_ptr.tobytes() == self.tap_ptr
                 and replay.tap_source_indices.tobytes() == self.tap_source_indices
                 and replay.tap_weight_numerators.tobytes() == self.tap_weight_numerators,
                 'witness_proof_replay_mismatch')

    @property
    def payload_bytes(self):
        return sum(len(v) for v in (self.map_x, self.map_y, self.depth_valid, self.tap_ptr,
                                   self.tap_source_indices, self.tap_weight_numerators)) + len(self.record_json.encode())

    @classmethod
    def from_replayed_witness(cls, witness):
        return cls(witness.source_shape, witness.map_x.shape,
                   witness.map_x.tobytes(), witness.map_y.tobytes(), witness.depth_valid.tobytes(),
                   witness.tap_ptr.tobytes(), witness.tap_source_indices.tobytes(),
                   witness.tap_weight_numerators.tobytes(), _witness_record_json(witness),
                   witness.witness_sha256)

    def sample_payload(self, index):
        _require(type(index) is int and 0 <= index < self.map_shape[0] * self.map_shape[1],
                 'sample_index_binding_mismatch')
        pointers = np.frombuffer(self.tap_ptr, dtype=np.int64)
        first, last = pointers[index:index + 2]
        indices = np.frombuffer(self.tap_source_indices, dtype=np.int64)[first:last]
        weights = np.frombuffer(self.tap_weight_numerators, dtype=np.uint16)[first:last]
        return ((float(np.frombuffer(self.map_x, dtype=np.float32)[index]),
                 float(np.frombuffer(self.map_y, dtype=np.float32)[index])),
                bool(self.depth_valid[index]),
                tuple((int(i), int(w)) for i, w in zip(indices, weights)))


def _witness_record_json(witness):
    return json.dumps(witness.to_record(), sort_keys=True, separators=(',', ':'), allow_nan=False)


def _witness_payload_bytes(witness):
    return sum(a.nbytes for a in (witness.map_x, witness.map_y, witness.depth_valid, witness.tap_ptr,
                                 witness.tap_source_indices, witness.tap_weight_numerators)) + len(_witness_record_json(witness).encode())


def _support_pixels(partition, samples, groups, witnesses):
    _require(type(partition) is SourceTapPartition, 'invalid_source_partition')
    _require(type(samples) is tuple and len(samples) <= LIMITS.max_rectified_samples
             and all(type(s) is TapSample for s in samples), 'invalid_support_samples')
    _require(type(groups) is tuple and all(type(g) is DependencyGroup for g in groups),
             'invalid_support_groups')
    _require(sum(len(g.sample_refs) for g in groups) <= LIMITS.max_dependency_memberships,
             'dependency_cap_exceeded')
    _require(sum(len(s.taps) for s in samples) <= LIMITS.max_bilinear_taps, 'tap_cap_exceeded')
    lookup = {(s.witness_index, s.sample_index): s for s in samples}
    _require(len(lookup) == len(samples), 'duplicate_tap_identity')
    # Every direct construction must join to the original snapshot, not merely
    # to a newly synthesized witness of the selected coordinates.
    _require(type(witnesses) is tuple and witnesses
             and all(type(w) is WitnessProof for w in witnesses), 'witness_proof_unavailable')
    _require(sum(w.map_shape[0] * w.map_shape[1] for w in witnesses) <= LIMITS.max_rectified_samples
             and sum(w.map_shape[0] * w.map_shape[1] * 4 for w in witnesses) <= LIMITS.max_bilinear_taps,
             'witness_proof_cap_exceeded')
    _require(sum(w.payload_bytes for w in witnesses) <= LIMITS.max_witness_payload_bytes,
             'witness_payload_cap_exceeded')
    for sample in samples:
        _require(sample.witness_index < len(witnesses), 'sample_witness_binding_mismatch')
        original = witnesses[sample.witness_index]
        _require(original.source_shape == partition.source_shape,
                 'witness_source_shape_mismatch')
        _require(sample.witness_sha256 == original.witness_sha256, 'sample_witness_binding_mismatch')
        source_map, depth, taps = original.sample_payload(sample.sample_index)
        _require(sample.source_map == source_map, 'sample_map_binding_mismatch')
        _require(sample.depth_valid == depth, 'sample_depth_binding_mismatch')
        _require(sample.taps == taps, 'sample_tap_replay_mismatch')
    _require(len({g.operation_id for g in groups}) == len(groups), 'duplicate_operation_identity')
    selected = set()
    pixels = {'generation': set(), 'validation': set()}
    for group in groups:
        for ref in group.sample_refs:
            _require(ref in lookup, 'sample_reference_unavailable')
            selected.add(ref)
            sample = lookup[ref]
            _require(sample.depth_valid, 'invalid_depth_sample')
            for index, _ in sample.taps:
                _require(index >= 0, 'border_source_tap')
                _require(index < len(partition.roles), 'source_tap_out_of_range')
                pixels[group.role].add(index)
    _require(selected == set(lookup), 'undeclared_sample_dependency')
    _require(not pixels['generation'] & pixels['validation'], 'overlapping_source_taps')
    for role, expected in (('generation', 1), ('validation', 2)):
        _require(pixels[role], 'missing_support_role')
        for index in pixels[role]:
            _require(partition.usable[index] and partition.region[index], 'excluded_source_tap')
            _require(partition.roles[index] == expected, 'cross_partition_stencil')
    return tuple(sorted(pixels['generation'])), tuple(sorted(pixels['validation']))


@dataclass(frozen=True, slots=True)
class PartitionedSupport:
    status: str
    reason: str
    partition: SourceTapPartition | None = None
    samples: tuple[TapSample, ...] = ()
    groups: tuple[DependencyGroup, ...] = ()
    witnesses: tuple[WitnessProof, ...] = ()
    generation_pixels: tuple[int, ...] = field(init=False, default=())
    validation_pixels: tuple[int, ...] = field(init=False, default=())

    def __post_init__(self):
        _require(self.status in ('complete', 'refused'), 'invalid_support_status')
        if self.status == 'refused':
            _require(type(self.reason) is str and self.reason and self.partition is None
                     and self.samples == () and self.groups == () and self.witnesses == (), 'non_atomic_support_refusal')
        else:
            _require(self.reason == '', 'complete_support_has_reason')
            g, v = _support_pixels(self.partition, self.samples, self.groups, self.witnesses)
            object.__setattr__(self, 'generation_pixels', g)
            object.__setattr__(self, 'validation_pixels', v)

    @property
    def identity_sha256(self):
        return _digest(asdict(self))


def partition_tap_support(partition, witnesses, groups, *, limits=LIMITS):
    """Replay exact tap witnesses, snapshot dependencies, or atomically refuse.

    Preflight counts precede witness replay. This does not call rectify_plane,
    read any image pixels, or treat a sink refusal as usable evidence.
    """
    try:
        _require(type(partition) is SourceTapPartition and type(limits) is ContractLimits,
                 'invalid_support_input')
        _require(type(witnesses) in (tuple, list) and type(groups) in (tuple, list),
                 'invalid_support_input')
        _require(0 < len(witnesses) <= limits.max_rectified_samples,
                 'sample_cap_exceeded')
        groups = tuple(groups)
        _require(all(type(g) is DependencyGroup for g in groups), 'invalid_support_groups')
        _require(sum(len(g.sample_refs) for g in groups) <= limits.max_dependency_memberships,
                 'dependency_cap_exceeded')
        total_samples = total_taps = total_proof_bytes = 0
        for witness in witnesses:
            _require(type(witness) is lattice.RectificationWitnessV1, 'witness_unavailable')
            _require(witness.source_shape == partition.source_shape, 'witness_source_shape_mismatch')
            _require(isinstance(witness.map_x, np.ndarray) and witness.map_x.ndim == 2,
                     'invalid_witness_maps')
            total_samples += witness.map_x.size
            total_taps += witness.map_x.size * 4
            total_proof_bytes += _witness_payload_bytes(witness)
        _require(len(partition.roles) <= limits.max_source_pixels, 'source_pixel_cap_exceeded')
        _require(total_samples <= limits.max_rectified_samples, 'sample_cap_exceeded')
        _require(total_taps <= limits.max_bilinear_taps, 'tap_cap_exceeded')
        _require(total_proof_bytes <= limits.max_witness_payload_bytes, 'witness_payload_cap_exceeded')
        requested = {ref for group in groups for ref in group.sample_refs}
        for wi, si in requested:
            _require(wi < len(witnesses) and si < witnesses[wi].map_x.size,
                     'sample_reference_unavailable')
        snapshots = []
        proofs = []
        for wi, witness in enumerate(witnesses):
            _require(all(isinstance(a, np.ndarray) and not a.flags.writeable for a in (
                witness.map_x, witness.map_y, witness.depth_valid, witness.tap_ptr,
                witness.tap_source_indices, witness.tap_weight_numerators)), 'mutable_witness')
            _require(witness.map_x.dtype == np.float32 and witness.map_y.dtype == np.float32
                     and witness.depth_valid.dtype == np.bool_
                     and witness.map_x.shape == witness.map_y.shape == witness.depth_valid.shape
                     and np.isfinite(witness.map_x).all() and np.isfinite(witness.map_y).all(),
                     'invalid_witness_maps')
            rebuilt = lattice.rectification_witness_v1(
                witness.map_x, witness.map_y, witness.depth_valid,
                source_shape=partition.source_shape,
                max_samples=limits.max_rectified_samples,
                max_taps=limits.max_bilinear_taps)
            _require(rebuilt.witness_sha256 == witness.witness_sha256 and
                     rebuilt.to_record() == witness.to_record(), 'witness_replay_mismatch')
            proofs.append(WitnessProof.from_replayed_witness(rebuilt))
            for si in sorted(index for idx, index in requested if idx == wi):
                first, last = rebuilt.tap_ptr[si:si + 2]
                taps = tuple((int(index), int(weight)) for index, weight in zip(
                    rebuilt.tap_source_indices[first:last], rebuilt.tap_weight_numerators[first:last]))
                snapshots.append(TapSample(wi, si, rebuilt.witness_sha256,
                    (float(rebuilt.map_x.flat[si]), float(rebuilt.map_y.flat[si])),
                    bool(rebuilt.depth_valid.flat[si]), taps))
        return PartitionedSupport('complete', '', partition, tuple(snapshots), groups, tuple(proofs))
    except (UnitFaceContractError, ValueError, TypeError, IndexError, OverflowError) as error:
        reason = error.reason if isinstance(error, UnitFaceContractError) else 'invalid_tap_evidence'
        return PartitionedSupport('refused', reason)


@dataclass(frozen=True, slots=True)
class RoleObservation:
    role: str
    support_sha256: str
    source_indices: tuple[int, ...]
    rgb: bytes

    def __post_init__(self):
        _require(type(self.role) is str and self.role in ('generation', 'validation') and _hex(self.support_sha256),
                 'invalid_observation_identity')
        _require(type(self.source_indices) is tuple and self.source_indices
                 and all(type(i) is int and i >= 0 for i in self.source_indices)
                 and tuple(sorted(set(self.source_indices))) == self.source_indices
                 and len(self.source_indices) <= LIMITS.max_source_pixels,
                 'invalid_observation_indices')
        _require(type(self.rgb) is bytes and len(self.rgb) == 3 * len(self.source_indices),
                 'invalid_observation_payload')

    @property
    def identity_sha256(self):
        return _digest(asdict(self))


def observe_role_pixels(image, support, role):
    """Snapshot only declared original-source support, without fitting anything."""
    _require(type(support) is PartitionedSupport and support.status == 'complete',
             'support_unavailable')
    _require(role in ('generation', 'validation'), 'invalid_observation_role')
    _require(isinstance(image, np.ndarray) and image.dtype == np.uint8
             and image.shape == support.partition.source_shape + (3,), 'invalid_source_image')
    indices = support.generation_pixels if role == 'generation' else support.validation_pixels
    rgb = image.reshape(-1, 3)[list(indices)].tobytes()
    return RoleObservation(role, support.identity_sha256, indices, rgb)


@dataclass(frozen=True, slots=True)
class UnitAlias:
    alias_id: str
    face_sha256: str
    blocks_per_period: float
    distance_interval: tuple[float, float]
    phase_u: tuple[float, float] | None  # start in [0,1), circular width <= 1
    phase_v: tuple[float, float] | None

    def __post_init__(self):
        _require(type(self.alias_id) is str and 0 < len(self.alias_id) <= 256 and _hex(self.face_sha256),
                 'invalid_alias_identity')
        _require(_finite(self.blocks_per_period) and self.blocks_per_period > 0, 'invalid_alias_scale')
        interval = self.distance_interval
        _require(type(interval) in (tuple, list) and len(interval) == 2
                 and all(_finite(v) for v in interval) and 0 < interval[0] <= interval[1],
                 'invalid_distance_interval')
        object.__setattr__(self, 'distance_interval', tuple(interval))
        for name in ('phase_u', 'phase_v'):
            phase = getattr(self, name)
            if phase is not None:
                _require(type(phase) in (tuple, list) and len(phase) == 2
                         and all(_finite(v) for v in phase)
                         and 0 <= phase[0] < 1 and 0 <= phase[1] <= 1, 'invalid_phase_interval')
                object.__setattr__(self, name, tuple(phase))


@dataclass(frozen=True, slots=True)
class UnitFaceLatticeResult:
    """Provisional DECLARATIVE outcome invariants, not an anchor estimator.

    Numeric fitting/residual/separation/error-bound certificates remain absent.
    Callers cannot promote this record into global camera scale or placement.
    """

    status: str
    period_measured: bool
    unit_scale_supported: bool
    phase_supported: bool
    placement_supported: bool
    abstention: str
    aliases: tuple[UnitAlias, ...]
    catalogue: UnitFaceCatalogue
    support: PartitionedSupport
    search_complete: bool
    validation_scope: str = field(init=False, default='conditional_on_frozen_upstream_context')
    reconstruction_authorized: bool = field(init=False, default=False)
    camera_scale_known: bool = field(init=False, default=False)

    def __post_init__(self):
        _require(self.status in ('supported', 'ambiguous', 'abstained'), 'invalid_result_status')
        for name in ('period_measured', 'unit_scale_supported', 'phase_supported',
                     'placement_supported', 'search_complete'):
            _require(type(getattr(self, name)) is bool, 'invalid_result_flag')
        _require(type(self.catalogue) is UnitFaceCatalogue and type(self.support) is PartitionedSupport,
                 'invalid_result_evidence')
        _require(type(self.aliases) in (tuple, list) and len(self.aliases) <= LIMITS.max_aliases
                 and all(type(a) is UnitAlias for a in self.aliases), 'invalid_result_aliases')
        aliases = tuple(self.aliases)
        _require(len({a.alias_id for a in aliases}) == len(aliases), 'duplicate_alias_identity')
        object.__setattr__(self, 'aliases', aliases)
        _require(type(self.abstention) is str and len(self.abstention) <= 256, 'invalid_abstention')
        _require(not self.placement_supported, 'placement_adapter_unavailable')
        _require(not self.phase_supported or self.unit_scale_supported, 'phase_without_unit_scale')
        _require(not self.unit_scale_supported or self.period_measured, 'scale_without_period')
        _require(not aliases or self.period_measured, 'aliases_without_period')
        if not self.search_complete or not self.catalogue.complete:
            _require(self.status == 'abstained' and self.abstention == 'search_incomplete'
                     and not aliases and not self.unit_scale_supported and not self.phase_supported,
                     'truncated_result_not_atomic')
            return
        if self.status == 'supported':
            _require(self.abstention == '' and self.unit_scale_supported and len(aliases) == 1
                     and self.support.status == 'complete', 'invalid_supported_result')
            faces = {f.identity_sha256: f for f in self.catalogue.faces}
            _require(aliases[0].face_sha256 in faces and
                     not faces[aliases[0].face_sha256].unavailable_reasons,
                     'unit_anchor_unavailable')
            if self.phase_supported:
                _require(all(p is not None and p[1] < 1 for p in (aliases[0].phase_u, aliases[0].phase_v)),
                         'phase_evidence_unavailable')
        elif self.status == 'ambiguous':
            _require(self.support.status == 'complete', 'ambiguity_without_source_support')
            _require(len(aliases) >= 2 and self.abstention == 'ambiguous_unit_alias'
                     and not self.unit_scale_supported and not self.phase_supported,
                     'invalid_ambiguous_result')
        else:
            _require(self.abstention and not self.unit_scale_supported and not self.phase_supported,
                     'invalid_abstained_result')
            if self.support.status == 'refused':
                _require(not aliases and self.abstention == 'support_unavailable',
                         'invalid_support_abstention')
        if self.status != 'supported' and aliases:
            identities = {f.identity_sha256 for f in self.catalogue.faces}
            _require(all(a.face_sha256 in identities for a in aliases), 'unknown_alias_template')

    @property
    def identity_sha256(self):
        return _digest(asdict(self))
