"""Private RGB-only adapter; calibration and proposals condition both fit roles.

The held-out veto is conditional on frozen image-derived upstream geometry,
not independent calibration evidence. No occupancy, placement or export follows.
Original parent pixels are bound through an integer native crop, never resized.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json

import numpy as np

from ..reconstruction import calibration, lattice, region_growth
from ..reconstruction.calibration import CalibrationResult
from ..reconstruction.unit_face_estimator import (
    FitResult, PlaneContext, SearchPolicy, context_from_calibration, estimate_unit_face,
)
from ..reconstruction.unit_face_lattice import (
    ContractLimits, DependencyGroup, PartitionedSupport, SourceTapPartition,
    UnitFaceContractError, WitnessProof, partition_tap_support,
)
from ..vision.gui_mask import create_gui_mask


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _digest(value):
    return _sha(json.dumps(value, sort_keys=True, separators=(',', ':'),
                           allow_nan=False).encode())


class PreparationError(ValueError):
    """Stable preparation refusal, before any numerical fit."""

    def __init__(self, reason, counts=None):
        self.reason = reason
        self.counts = counts
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class InputPolicy:
    """Lowerable preparation/work caps only; no optics or distance inputs.

    The fixed crop is at most 256 square pixels, sampling is endpoint-inclusive,
    and roles use absolute-parent 8-pixel checker tiles before any fitting.
    """

    max_frame_pixels: int = 6_000_000
    max_crop_pixels: int = 65_536
    max_proposals: int = 32
    samples_per_side: int = 20
    max_source_pixels_per_role: int = 2048
    max_witness_samples: int = 4096
    max_witness_taps: int = 16384
    max_dependency_memberships: int = 8192
    max_witness_bytes: int = 1_048_576
    max_fit_work: int = 100_000
    max_fit_pixel_work: int = 250_000_000

    def __post_init__(self):
        ceilings = (6_000_000, 65_536, 32, 20, 2048, 4096, 16384,
                    8192, 1_048_576, 100_000, 250_000_000)
        if any(type(v) is not int or not 0 < v <= cap
               for v, cap in zip(asdict(self).values(), ceilings)) or self.samples_per_side < 2:
            raise ValueError('invalid_input_policy')

    @property
    def sha256(self):
        return _digest({'policy': asdict(self), 'crop_side': 256, 'role_tile': 8,
                        'selection': 'strength-area-axis-side-mask-v1', 'window_margin': 1.05,
                        'support': 'complete-nonfinal-row-column-taps-v1'})


@dataclass(frozen=True, slots=True)
class RgbFrame:
    frame_id: str
    shape: tuple[int, int]
    rgb: bytes
    sha256: str

    @classmethod
    def snapshot(cls, image, frame_id, policy):
        if (type(image) is not np.ndarray or image.dtype != np.uint8
                or image.ndim != 3 or image.shape[2] != 3
                or not image.shape[0] or not image.shape[1]
                or type(frame_id) is not str or not 0 < len(frame_id) <= 256):
            raise PreparationError('invalid_rgb_frame')
        if image.shape[0] * image.shape[1] > policy.max_frame_pixels:
            raise PreparationError('frame_pixel_cap')
        payload = image.tobytes(order='C')
        shape = tuple(image.shape[:2])
        return cls(frame_id, shape, payload,
                   _digest({'frame_id': frame_id, 'shape': shape, 'rgb_sha256': _sha(payload)}))

    def image(self):
        return np.frombuffer(self.rgb, dtype=np.uint8).reshape(self.shape + (3,))


@dataclass(frozen=True, slots=True)
class ProposalSnapshot:
    axis: int
    side: int
    polygon: tuple[tuple[float, float], ...]
    strength: float
    sources: tuple[str, ...]
    coverage: float
    mask: bytes
    sha256: str


@dataclass(frozen=True, slots=True)
class PreparationCounts:
    proposals: int = 0
    crop_pixels: int = 0
    region_pixels: int = 0
    witness_samples: int = 0
    retained_samples: int = 0
    rejected_depth: int = 0
    rejected_border: int = 0
    rejected_excluded: int = 0
    rejected_cross_role: int = 0
    generation_pixels: int = 0
    validation_pixels: int = 0
    witness_bytes: int = 0


@dataclass(frozen=True, slots=True)
class PreparedInput:
    frame: RgbFrame
    full_calibration: CalibrationResult
    proposal: ProposalSnapshot
    proposal_order: tuple[str, ...]
    crop: tuple[int, int, int, int]  # x0,y0,width,height; integer translation
    crop_rgb: bytes
    cropped_calibration: CalibrationResult
    context: PlaneContext
    support: PartitionedSupport
    policy: InputPolicy
    counts: PreparationCounts
    binding_sha256: str

    def image(self):
        return np.frombuffer(self.crop_rgb, dtype=np.uint8).reshape((self.crop[3], self.crop[2], 3))

    def parent_pixels(self, role):
        """Replay crop-owned source indices into the original RGB frame."""
        if role not in ('generation', 'validation'):
            raise ValueError('invalid_role')
        x0, y0, width, _ = self.crop
        return tuple((p // width + y0) * self.frame.shape[1] + p % width + x0
                     for p in getattr(self.support, role + '_pixels'))


@dataclass(frozen=True, slots=True)
class ImageUnitFaceResult:
    status: str
    reason: str
    prepared: PreparedInput | None = None
    fit: FitResult | None = None
    counts: PreparationCounts = PreparationCounts()
    calibration_status: str = 'not_run'
    proposal_status: str = 'not_run'
    placement_supported: bool = False
    validation_interpretation: str = 'conditional_on_frozen_upstream_image_context'
    full_calibration: CalibrationResult | None = None


def _proposal_record(proposal, shape):
    mask = proposal.mask
    if (type(mask) is not np.ndarray or mask.dtype != np.bool_ or mask.shape != shape
            or proposal.axis not in (0, 1, 2) or proposal.side not in (-1, 1)
            or not np.isfinite((proposal.strength, proposal.coverage)).all()):
        raise PreparationError('invalid_image_proposal')
    values = dict(axis=proposal.axis, side=proposal.side,
                  polygon=tuple(tuple(float(v) for v in p) for p in proposal.polygon),
                  strength=float(proposal.strength), sources=tuple(proposal.sources),
                  coverage=float(proposal.coverage), mask=mask.tobytes())
    digest_values = dict(values, mask=_sha(values['mask']), shape=shape)
    return ProposalSnapshot(**values, sha256=_digest(digest_values))


def _select_proposal(proposals, shape, policy):
    if len(proposals) > policy.max_proposals:
        raise PreparationError('proposal_cap')
    snapshots = [_proposal_record(p, shape) for p in proposals]
    snapshots.sort(key=lambda p: (-p.strength, -p.mask.count(1), p.axis, p.side,
                                   _sha(p.mask), p.sha256))
    if not snapshots:
        raise PreparationError('no_image_proposals')
    return snapshots[0], tuple(p.sha256 for p in snapshots)


def _crop_for(proposal, shape, policy):
    ys, xs = np.nonzero(np.frombuffer(proposal.mask, np.bool_).reshape(shape))
    if not len(xs):
        raise PreparationError('empty_image_proposal')
    width, height = min(256, int(xs.max() - xs.min() + 1)), min(256, int(ys.max() - ys.min() + 1))
    x0 = int(xs.min()) + (int(xs.max() - xs.min() + 1) - width) // 2
    y0 = int(ys.min()) + (int(ys.max() - ys.min() + 1) - height) // 2
    if width * height > policy.max_crop_pixels:
        raise PreparationError('crop_pixel_cap')
    return x0, y0, width, height


def _conditioning(frame, cal, proposal, order, crop, policy):
    return _digest({'parent_image': frame.sha256, 'full_calibration': asdict(cal),
                    'proposal': proposal.sha256, 'proposal_order': order,
                    'crop': crop, 'policy': policy.sha256})


def _binding(prepared):
    return _digest({'conditioning': _conditioning(prepared.frame, prepared.full_calibration,
                        prepared.proposal, prepared.proposal_order, prepared.crop, prepared.policy),
                    'crop_rgb': _sha(prepared.crop_rgb),
                    'cropped_calibration': asdict(prepared.cropped_calibration),
                    'context': prepared.context.sha256,
                    'support': prepared.support.identity_sha256, 'counts': asdict(prepared.counts)})


def _prepare(frame, cal, proposal, order, policy):
    """Internal seam consumes only outputs of the actual image stages."""
    if type(cal) is not CalibrationResult or (cal.height, cal.width) != frame.shape:
        raise PreparationError('invalid_full_frame_calibration')
    crop = _crop_for(proposal, frame.shape, policy)
    x0, y0, width, height = crop
    crop_image = frame.image()[y0:y0 + height, x0:x0 + width]
    full_region = np.frombuffer(proposal.mask, np.bool_).reshape(frame.shape)
    region = full_region[y0:y0 + height, x0:x0 + width].copy()
    usable = (~create_gui_mask(*frame.shape))[y0:y0 + height, x0:x0 + width]
    pp = cal.principal_point or (cal.width / 2, cal.height / 2)
    cropped = replace(cal, width=width, height=height, principal_point=(pp[0] - x0, pp[1] - y0))
    conditioning = _conditioning(frame, cal, proposal, order, crop, policy)
    context = context_from_calibration(cropped, proposal.axis, conditioning_sha256=conditioning)
    yy, xx = np.indices((height, width))
    generation = usable & region & (((xx + x0) // 8 + (yy + y0) // 8) % 2 == 0)
    validation = usable & region & ~generation
    partition = SourceTapPartition.from_masks(generation, validation, usable, region,
                                             conditioning_sha256=conditioning)
    samples = policy.samples_per_side ** 2
    preliminary = PreparationCounts(proposals=len(order), crop_pixels=width*height,
                                     region_pixels=int(region.sum()), witness_samples=samples)
    if samples > policy.max_witness_samples:
        raise PreparationError('witness_sample_cap', preliminary)
    if samples * 4 > policy.max_witness_taps:
        raise PreparationError('witness_tap_cap', preliminary)
    matrix = np.asarray(context.axis_matrix)
    window = lattice.plane_window(region & usable, context.axis, matrix, context.focal_pixels,
                                  principal_point=context.principal_point)
    if window is None:
        raise PreparationError('unrectifiable_image_region')
    if window[3] != proposal.side:
        raise PreparationError('proposal_side_inconsistent')
    witnesses = []
    lattice.rectify_plane(crop_image, context.axis, matrix, context.focal_pixels,
                         extent=window[2], centre=window[:2], side=window[3],
                         samples=policy.samples_per_side, principal_point=context.principal_point,
                         witness_sink=witnesses)
    if len(witnesses) != 1 or type(witnesses[0]) is not lattice.RectificationWitnessV1:
        raise PreparationError('rectification_witness_unavailable')
    witness = witnesses[0]
    groups = []
    role_pixels = {'generation': set(), 'validation': set()}
    rejected = dict(depth=0, border=0, excluded=0, cross_role=0)
    for index in range(samples):
        first, last = witness.tap_ptr[index:index + 2]
        pixels = witness.tap_source_indices[first:last]
        if not witness.depth_valid.flat[index]:
            rejected['depth'] += 1
        elif (np.any(pixels < 0) or any(int(p) % width == width-1 or int(p) // width == height-1
                                      for p in pixels)):
            rejected['border'] += 1
        elif any(not partition.usable[int(p)] or not partition.region[int(p)] for p in pixels):
            rejected['excluded'] += 1
        else:
            roles = {partition.roles[int(p)] for p in pixels}
            if roles not in ({1}, {2}):
                rejected['cross_role'] += 1
                continue
            role = 'generation' if roles == {1} else 'validation'
            groups.append(DependencyGroup(f'original-rgb-{index}', role, ((0, index),)))
            role_pixels[role].update(int(p) for p in pixels)
    proof_bytes = WitnessProof.from_replayed_witness(witness).payload_bytes
    counts = PreparationCounts(len(order), width * height, int(region.sum()), samples, len(groups),
        rejected['depth'], rejected['border'], rejected['excluded'], rejected['cross_role'],
        len(role_pixels['generation']), len(role_pixels['validation']), proof_bytes)
    limits = ContractLimits(max_source_pixels=policy.max_crop_pixels,
        max_rectified_samples=policy.max_witness_samples, max_bilinear_taps=policy.max_witness_taps,
        max_dependency_memberships=policy.max_dependency_memberships,
        max_witness_payload_bytes=policy.max_witness_bytes)
    support = partition_tap_support(partition, witnesses, groups, limits=limits)
    if support.status != 'complete':
        raise PreparationError(support.reason, counts)
    if max(len(support.generation_pixels), len(support.validation_pixels)) > policy.max_source_pixels_per_role:
        raise PreparationError('role_source_pixel_cap', counts)
    if min(len(support.generation_pixels), len(support.validation_pixels)) < 48:
        raise PreparationError('insufficient_role_source_pixels', counts)
    prepared = PreparedInput(frame, cal, proposal, order, crop, crop_image.tobytes(), cropped,
                             context, support, policy, counts, '')
    return replace(prepared, binding_sha256=_binding(prepared))


def _verify_prepared(prepared):
    """Reconstruct snapshots and geometry; never rerun/choose upstream on fit RGB."""
    try:
        if type(prepared) is not PreparedInput or prepared.binding_sha256 != _binding(prepared):
            raise PreparationError('stale_preparation_binding')
        frame = RgbFrame.snapshot(prepared.frame.image(), prepared.frame.frame_id, prepared.policy)
        if frame != prepared.frame:
            raise PreparationError('stale_parent_image_binding')
        p = prepared.proposal
        record = asdict(p)
        record.pop('sha256')
        record['mask'] = _sha(p.mask)
        record['shape'] = frame.shape
        if p.sha256 != _digest(record):
            raise PreparationError('stale_proposal_binding')
        # The unchanged estimator also replays all original witness payloads.
        for witness in prepared.support.witnesses:
            WitnessProof(**asdict(witness))
        rebuilt = _prepare(frame, prepared.full_calibration, p, prepared.proposal_order, prepared.policy)
        if rebuilt != prepared:
            raise PreparationError('stale_geometry_or_source_binding')
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, UnitFaceContractError) as error:
        if isinstance(error, PreparationError):
            raise
        raise PreparationError('invalid_preparation_binding') from error


def _fit_prepared(prepared, catalogue):
    _verify_prepared(prepared)
    policy = replace(SearchPolicy(), max_work=prepared.policy.max_fit_work,
                     max_pixel_work=prepared.policy.max_fit_pixel_work,
                     max_pixels_per_role=prepared.policy.max_source_pixels_per_role)
    return estimate_unit_face(prepared.image(), prepared.context, prepared.support, catalogue,
                              fixed_policy=policy)


def estimate_image_unit_face(image, catalogue, *, frame_id='frame', policy=InputPolicy()):
    """Only RGB/catalogue and lowerable caps enter the actual image-derived path.

    There are no arguments for manifests, masks, palettes, F3, axes, lengths,
    camera optics, distances, bounds or placement. Select exactly once before
    fitting; validation cannot choose another crop/proposal/template.
    """
    cal_status = proposal_status = 'not_run'
    cal = None
    counts = PreparationCounts()
    try:
        if type(policy) is not InputPolicy:
            raise PreparationError('invalid_input_policy')
        frame = RgbFrame.snapshot(image, frame_id, policy)
        cal = calibration.estimate_image_calibrations({frame_id: frame.image()})[frame_id]
        cal_status = 'failed' if cal is None else 'complete'
        if cal is None:
            raise PreparationError('image_calibration_failed')
        proposals = region_growth.propose_image_planes_v2(frame.image(),
            ~create_gui_mask(*frame.shape), cal, max_candidates=32)
        proposal_status = proposals.status
        if proposals.status != 'ok':
            raise PreparationError('image_proposal_' + proposals.reason)
        proposal, order = _select_proposal(proposals.proposals, frame.shape, policy)
        counts = PreparationCounts(proposals=len(order))
        prepared = _prepare(frame, cal, proposal, order, policy)
        counts = prepared.counts
        fit = _fit_prepared(prepared, catalogue)
        return ImageUnitFaceResult(fit.status, fit.reason, prepared, fit, counts,
                                   cal_status, proposal_status, full_calibration=cal)
    except (PreparationError, UnitFaceContractError, ValueError) as error:
        reason = getattr(error, 'reason', str(error))
        return ImageUnitFaceResult('abstained', reason, counts=getattr(error, 'counts', None) or counts,
                                   calibration_status=cal_status, proposal_status=proposal_status,
                                   full_calibration=cal)
