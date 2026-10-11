"""Isolated opt-in exhaustive appearance diagnostics with atomic logical caps.

Admission precedes replay, hashing, scorer construction and vector allocation.
Array copies made by FaceObservation before entry are outside this contract.
Logical accounting is not a bound on measured RSS or execution time.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields, replace

import numpy as np

from mc_photogrammetry.blocks.appearance import PaletteColor
from mc_photogrammetry.blocks.candidate_universe import (
    AppearanceSupport,
    CandidateUniverse,
    _preflight,
    canonical_state_id,
    record_sha256,
)
from mc_photogrammetry.blocks.model_atlas import FACE_SLOTS
from mc_photogrammetry.domain.models import BlockState
from mc_photogrammetry.pipeline.image_derived_output_policy_v2 import (
    ImageDerivedOutputPolicyV2,
    ProposalIdentity,
    ScorerIdentity,
    _policy_preflight,
)

from .appearance_scoring_v2 import (
    AppearanceScorerV2,
    DiagnosticScores,
    aggregate_scores,
    cost_bits,
    cost_from_bits,
    diagnostic_scores,
    finite,
    observation_features,
    scorer_definition,
    scorer_identity,
)
from .face_observation import NORMAL_OF_SLOT, FaceObservation

RESULT_SCHEMA = "mc-photogrammetry-exhaustive-appearance-diagnostics-v2"
OBSERVATION_SCHEMA = "mc-photogrammetry-bounded-face-observation-v2"
MAX_CANDIDATES = 16_384
MAX_OBSERVATIONS = 1000
MAX_EVALUATIONS = 100_000
MAX_PATCH_PIXELS = 4096
MAX_TOTAL_PIXELS = 1_000_000
MAX_OBSERVATION_ARRAY_BYTES = 86_016
MAX_TOTAL_ARRAY_BYTES = 21_000_000
MAX_PIXEL_WORK = 25_600_000
MAX_DESCRIPTOR_WORK = 4_200_000
MAX_STRING_BYTES = 1024
MAX_REFERENCES = 32
MAX_OBSERVATION_METADATA_BYTES = 16_384
MAX_METADATA_BYTES = 1_000_000
MAX_CELL_PAYLOAD_BYTES = 8_000_000
MAX_PAYLOAD_BYTES = 10_000_000
MAX_BASE_CHAIN = 8


def producer_resource_profile() -> dict:
    return {
        "version": 1, "candidates": MAX_CANDIDATES, "observations": MAX_OBSERVATIONS,
        "candidate_evaluations": MAX_EVALUATIONS, "height": [2, 256], "width": [1, 256],
        "pixels_per_patch": MAX_PATCH_PIXELS, "patch_pixels": MAX_TOTAL_PIXELS,
        "array_bytes_per_observation": MAX_OBSERVATION_ARRAY_BYTES,
        "array_bytes": MAX_TOTAL_ARRAY_BYTES, "candidate_pixel_work": MAX_PIXEL_WORK,
        "descriptors_per_candidate": 6, "descriptor_width": 7,
        "descriptor_component_magnitude": 1e6, "descriptor_work": MAX_DESCRIPTOR_WORK,
        "face_rgb_map_slots": 6, "observation_weight": [0, 1e6],
        "string_utf8_bytes": MAX_STRING_BYTES, "references_per_kind": MAX_REFERENCES,
        "metadata_bytes_per_observation": MAX_OBSERVATION_METADATA_BYTES,
        "metadata_bytes": MAX_METADATA_BYTES, "cell_payload_bytes": MAX_CELL_PAYLOAD_BYTES,
        "payload_bytes": MAX_PAYLOAD_BYTES, "base_chain": MAX_BASE_CHAIN, "integer_bits": 64,
    }


def proposal_definition() -> dict:
    return {
        "id": "exhaustive-appearance-diagnostics-producer-v2", "version": "1",
        "scorer_definition_sha256": record_sha256(scorer_definition()),
        "resources": producer_resource_profile(),
        "completeness": (
            'exhaustive legal inventory and every disposition supported; '
            'otherwise atomic refusal'
        ),
        "identities": (
            'exact built-in fields and recomputed definition digests after '
            'bounded policy replay'
        ),
        "preflight": ["exact types/caps/count products",
                      "bounded metadata/array headers/candidates",
                      "bounded policy and universe replay/completeness/exact identities",
                      "array values/duplicate identity/canonical metadata order",
                      "arithmetic worst-case output reservation",
                      "stream array hashes/features/scores",
                      "full indexed row validation/canonical serialization/digest"],
        "observation_schema": OBSERVATION_SCHEMA,
        "observation_encoding": (
            "all metadata; array absence/dtype.str/byteorder/shape/C-order bytes"
        ),
        "ordering": (
            'observations by (cell,slot,frame_id); references sorted unique; '
            'cells lexicographic'
        ),
        "output": (
            'state ID table once; implicit vector indices; all absolute/relative '
            'values; all exact tie groups including singletons'
        ),
        "cost_encoding": "16 lowercase big-endian IEEE-754 binary64 hex digits; positive zero",
        "no_evidence": "explicit reason, null vectors and empty ties; no invented scores",
        "ties": "finite absolute equality, groups ascending cost, indices ascending",
        "hashes": "SHA256 compact sorted-key ensure_ascii JSON; array data streamed by memoryview",
        "work": (
            'logical counters, aliases counted per occurrence, advanced only on '
            'fully committed units. Reserved: every admitted candidate and '
            'observation, the candidate_evaluations product, all patch pixels, '
            'retained array bytes, metadata and hash bytes, and the worst-case '
            'complete aggregate vectors. Executed observation unit: only after '
            'feature extraction and a complete validated scored/no-evidence row, '
            'add one observation with its patch pixels, retained and metadata '
            'bytes. '
            'Executed hash unit: add only the exact array bytes already streamed '
            'for a completed observation hash. Executed scoring unit: only a full '
            'finite candidate vector that is returned and validated adds the '
            'candidate count plus candidates*observations evaluations, '
            'candidates*observation patch pixels and admitted candidate descriptor '
            'components; a partial or aborted vector adds nothing. Executed '
            'aggregate unit: only a full validated cell vector adds the candidate '
            'count. Admission and pre-observation refusals commit no unit and keep '
            'executed work at zero; every later refusal keeps its truthful typed '
            'reserved/executed counters while still exposing no rows, no cells and '
            'only the fixed minimal refusal payload.'
        ),
        "reservation": (
            'canonical scalar sizes plus worst-case singleton ties and complete '
            'vectors; includes counters/digests/delimiters'
        ),
        "refusal": (
            'typed atomic no scored rows; minimal record omits inventory; no '
            'payload if minimum exceeds caller cap'
        ),
        "selection": (
            'absent; no state winner, flag, threshold, confidence, retrieval or '
            'truncation'
        ),
    }


def proposal_identity() -> ProposalIdentity:
    return ProposalIdentity(
        "exhaustive-appearance-diagnostics-producer-v2", "1",
        record_sha256(proposal_definition()), OBSERVATION_SCHEMA, "1",
        "complete exact finite equality partition including singleton indices",
    )


@dataclass(frozen=True, slots=True)
class Work:
    candidates: int = 0
    observations: int = 0
    candidate_evaluations: int = 0
    patch_pixels: int = 0
    retained_array_bytes: int = 0
    candidate_pixel_work: int = 0
    descriptor_components: int = 0
    metadata_bytes: int = 0
    hash_bytes: int = 0
    aggregate_components: int = 0


@dataclass(frozen=True, slots=True)
class ArrayDigest:
    dtype: str
    byteorder: str
    shape: tuple[int, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class ObservationMetadata:
    cell: tuple[int, int, int]
    slot: str
    frame_id: str
    normal: tuple[str, ...]
    usable_coverage: str
    weight: str
    support_frames: tuple[str, ...]
    provenance: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ObservationDiagnostic:
    index: int
    metadata: ObservationMetadata
    arrays: tuple[ArrayDigest | None, ...]
    input_sha256: str
    status: str
    reason: str | None
    effective_mass: float
    fusion_weight: float
    scores: DiagnosticScores | None


@dataclass(frozen=True, slots=True)
class CellDiagnostic:
    cell: tuple[int, int, int]
    observation_indices: tuple[int, ...]
    input_sha256: str
    status: str
    reason: str | None
    weight_mass: float
    scores: DiagnosticScores | None


@dataclass(frozen=True, slots=True)
class DiagnosticResult:
    status: str
    reason: str | None
    completeness: str
    state_ids: tuple[str, ...] = ()
    observations: tuple[ObservationDiagnostic, ...] = ()
    cells: tuple[CellDiagnostic, ...] = ()
    scorer: ScorerIdentity | None = None
    proposal: ProposalIdentity | None = None
    policy_sha256: str | None = None
    universe_sha256: str | None = None
    input_sha256: str | None = None
    reserved_work: Work = Work()
    executed_work: Work = Work()
    reserved_payload_bytes: int = 0
    payload: bytes | None = None
    payload_sha256: str | None = None

    def to_record(self) -> dict | None:
        return None if self.payload is None else json.loads(self.payload)


class _Refusal(ValueError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _limit(value: int, maximum: int, reason: str) -> None:
    if value > maximum:
        raise _Refusal(reason)


def _text(value: str) -> None:
    if type(value) is not str or not value or len(value) > MAX_STRING_BYTES:
        raise _Refusal("observation_string")
    _limit(len(value.encode("utf-8")), MAX_STRING_BYTES, "observation_string")


def _number(value, low: float, high: float) -> float:
    if type(value) not in (int, float, np.float32, np.float64):
        raise _Refusal("numeric_type")
    if type(value) is int and value.bit_length() > 64:
        raise _Refusal("integer_bits")
    result = finite(value)
    if not low <= result <= high:
        raise _Refusal("numeric_range")
    return result


def _metadata(observation: FaceObservation) -> ObservationMetadata:
    if (type(observation.cell) is not tuple or len(observation.cell) != 3
            or any(type(value) is not int or value.bit_length() > 64
                   for value in observation.cell)):
        raise _Refusal("cell_metadata")
    for value in (observation.slot, observation.frame_id):
        _text(value)
    if observation.slot not in FACE_SLOTS:
        raise _Refusal("slot_metadata")
    if type(observation.normal) is not tuple or len(observation.normal) != 3:
        raise _Refusal("normal_metadata")
    normal = tuple(_number(value, -1.0, 1.0) for value in observation.normal)
    if (abs(math.sqrt(sum(value * value for value in normal)) - 1.0) > 1e-6
            or any(abs(a - b) > 1e-6
                   for a, b in zip(normal, NORMAL_OF_SLOT[observation.slot]))):
        raise _Refusal("normal_metadata")
    for refs in (observation.support_frames, observation.provenance):
        if type(refs) is not tuple:
            raise _Refusal("reference_type")
        _limit(len(refs), MAX_REFERENCES, "reference_count")
        for value in refs:
            _text(value)
    if (not observation.support_frames
            or observation.frame_id not in observation.support_frames):
        raise _Refusal("support_metadata")
    usable = _number(observation.usable_coverage, 0.0, 1.0)
    weight = _number(observation.weight, 0.0, 1e6)
    return ObservationMetadata(
        observation.cell, observation.slot, observation.frame_id,
        tuple(cost_bits(value) for value in normal), cost_bits(usable), cost_bits(weight),
        observation.support_frames, observation.provenance,
    )


def _array_header(array: np.ndarray, shape: tuple[int, ...], kind: str, size: int) -> None:
    if (type(array) is not np.ndarray or array.shape != shape
            or array.dtype.kind != kind or array.dtype.itemsize != size
            or not array.flags.c_contiguous or array.flags.writeable
            or array.nbytes != math.prod(shape) * size):
        raise _Refusal("array_header")
    current = array
    for _ in range(MAX_BASE_CHAIN):
        if type(current) is bytes:
            if len(current) != array.nbytes:
                raise _Refusal("array_backing_size")
            return
        if (type(current) is not np.ndarray or current.flags.writeable
                or not current.flags.c_contiguous or current.nbytes != array.nbytes
                or any(type(value) is not int or not 0 <= value <= MAX_OBSERVATION_ARRAY_BYTES
                       for value in (*current.shape, *current.strides))):
            raise _Refusal("array_backing")
        current = current.base
    raise _Refusal("array_base_chain")


def _observation_array_bytes(observation: FaceObservation) -> int:
    """Admitted per-occurrence retained bytes: 12 RGB + 1 valid + 8 coverage per pixel."""
    pixels = math.prod(observation.patch.shape[:2])
    return pixels * (12 + (1 if observation.valid is not None else 0)
                     + (8 if observation.coverage is not None else 0))


def _state_header(state: BlockState) -> str:
    if type(state) is not BlockState:
        raise _Refusal("state_type")
    _text(state.name)
    if type(state.properties) is not tuple or len(state.properties) > 1024:
        raise _Refusal("state_properties")
    size = len(state.name.encode("utf-8"))
    for pair in state.properties:
        if type(pair) is not tuple or len(pair) != 2:
            raise _Refusal("state_properties")
        for value in pair:
            _text(value)
        size += len(pair[0].encode("utf-8")) + len(pair[1].encode("utf-8")) + 2
        _limit(size + 1, MAX_STRING_BYTES, "state_id_bytes")
    state_id = state.name
    if state.properties:
        state_id += "[" + ",".join(key + "=" + value for key, value in state.properties) + "]"
    _limit(len(state_id.encode("utf-8")), MAX_STRING_BYTES, "state_id_bytes")
    return state_id


def _candidate_header(entry: AppearanceSupport) -> int:
    if type(entry) is not AppearanceSupport:
        raise _Refusal("support_type")
    _text(entry.state_id)
    if entry.disposition != "supported":
        return 0
    appearance = entry.appearance
    if type(appearance) is not PaletteColor:
        raise _Refusal("appearance_type")
    if type(appearance.descriptors) is not tuple:
        raise _Refusal("descriptor_type")
    _limit(len(appearance.descriptors), 6, "descriptor_count")
    for descriptor in appearance.descriptors:
        if type(descriptor) is not tuple or len(descriptor) != 7:
            raise _Refusal("descriptor_shape")
        for value in descriptor:
            _number(value, -1e6, 1e6)
    if type(appearance.face_map) is not tuple:
        raise _Refusal("face_map_type")
    _limit(len(appearance.face_map), 6, "face_map_count")
    slots = []
    for pair in appearance.face_map:
        if type(pair) is not tuple or len(pair) != 2:
            raise _Refusal("face_map_pair")
        slot, rgb = pair
        if type(slot) is not str or slot not in FACE_SLOTS:
            raise _Refusal("face_map_slot")
        slots.append(slot)
        if (type(rgb) is not tuple or len(rgb) != 3
                or any(type(value) is not int or not 0 <= value <= 255 for value in rgb)):
            raise _Refusal("face_map_rgb")
    if len(set(slots)) != len(slots):
        raise _Refusal("face_map_duplicate")
    return len(appearance.descriptors) * 7


def _measure(record: dict, maximum: int = MAX_PAYLOAD_BYTES) -> int:
    return _preflight(record, max_bytes=maximum)[1]


def _admit(policy: ImageDerivedOutputPolicyV2, observations: tuple) -> tuple:
    # No arbitrary iterables, nested replay, pixels, hashes or scorer here.
    if type(policy) is not ImageDerivedOutputPolicyV2 or type(observations) is not tuple:
        raise _Refusal("top_level_type")
    for label, maximum in (("max_observations", MAX_OBSERVATIONS),
                           ("max_candidate_evaluations", MAX_EVALUATIONS),
                           ("max_proposal_payload_bytes", MAX_PAYLOAD_BYTES)):
        value = getattr(policy, label)
        if type(value) is not int or not 1 <= value <= maximum:
            raise _Refusal("policy_cap")
    universe = policy.universe
    if type(universe) is not CandidateUniverse:
        raise _Refusal("universe_type")
    if type(universe.legal_states) is not tuple or type(universe.appearance_support) is not tuple:
        raise _Refusal("inventory_type")
    count, observation_count = len(universe.legal_states), len(observations)
    if count == 0 or len(universe.appearance_support) != count:
        raise _Refusal("inventory_count")
    _limit(count, MAX_CANDIDATES, "candidates")
    _limit(observation_count, policy.max_observations, "observations")
    _limit(count * observation_count, policy.max_candidate_evaluations, "candidate_evaluations")
    metadata, pixels, retained, metadata_bytes = [], 0, 0, 0
    for observation in observations:
        if type(observation) is not FaceObservation:
            raise _Refusal("observation_type")
        meta = _metadata(observation)
        try:
            size = _measure(asdict(meta), MAX_OBSERVATION_METADATA_BYTES)
        except ValueError as error:
            raise _Refusal("observation_metadata_bytes") from error
        metadata_bytes += size
        _limit(metadata_bytes, MAX_METADATA_BYTES, "metadata_bytes")
        patch = observation.patch
        if type(patch) is not np.ndarray or patch.ndim != 3 or patch.shape[2] != 3:
            raise _Refusal("patch_shape")
        height, width = patch.shape[:2]
        if not 2 <= height <= 256 or not 1 <= width <= 256:
            raise _Refusal("patch_dimensions")
        patch_pixels = height * width
        _limit(patch_pixels, MAX_PATCH_PIXELS, "pixels_per_patch")
        _array_header(patch, (height, width, 3), "f", 4)
        size = patch_pixels * 12
        for array, kind, itemsize in ((observation.valid, "b", 1),
                                      (observation.coverage, "f", 8)):
            if array is not None:
                _array_header(array, (height, width), kind, itemsize)
                size += patch_pixels * itemsize
        _limit(size, MAX_OBSERVATION_ARRAY_BYTES, "array_bytes_per_observation")
        pixels += patch_pixels
        retained += size  # Repeated aliases count on every occurrence.
        _limit(pixels, MAX_TOTAL_PIXELS, "patch_pixels")
        _limit(retained, MAX_TOTAL_ARRAY_BYTES, "retained_array_bytes")
        _limit(count * pixels, MAX_PIXEL_WORK, "candidate_pixel_work")
        metadata.append(meta)
    state_ids = tuple(_state_header(state) for state in universe.legal_states)
    components = 0
    for entry in universe.appearance_support:
        components += _candidate_header(entry)
        _limit(observation_count * components, MAX_DESCRIPTOR_WORK, "descriptor_components")
    work = Work(count, observation_count, count * observation_count, pixels, retained,
                count * pixels, observation_count * components, metadata_bytes, retained, 0)
    return tuple(metadata), state_ids, work, components


def _validate_values(observation: FaceObservation) -> None:
    # Only admitted byte counts can reach these numeric scans.
    for array, ceiling in ((observation.patch, 255.0), (observation.coverage, 1.0)):
        if array is not None and (not np.isfinite(array).all()
                                  or bool(np.any(array < 0)) or bool(np.any(array > ceiling))):
            raise _Refusal("array_values")
    for refs in (observation.support_frames, observation.provenance):
        if refs != tuple(sorted(set(refs))):
            raise _Refusal("reference_order")


def _array_record(array: ArrayDigest | None) -> dict | None:
    return None if array is None else asdict(array)


def _hash_array(array: np.ndarray | None) -> ArrayDigest | None:
    if array is None:
        return None
    header = {"dtype": array.dtype.str, "byteorder": array.dtype.byteorder,
              "shape": list(array.shape), "order": "C"}
    digest = hashlib.sha256(_serialize(header))
    view = memoryview(array).cast("B")
    for offset in range(0, len(view), 65_536):
        digest.update(view[offset:offset + 65_536])
    return ArrayDigest(array.dtype.str, array.dtype.byteorder, array.shape, digest.hexdigest())


def _serialize(record: dict) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _score_record(scores: DiagnosticScores | None) -> dict:
    return {"absolute": None if scores is None else [cost_bits(v) for v in scores.absolute],
            "relative": None if scores is None else [cost_bits(v) for v in scores.relative],
            "ties": [] if scores is None else [list(group) for group in scores.ties]}


def _observation_record(row: ObservationDiagnostic) -> dict:
    return {"index": row.index, "metadata": asdict(row.metadata),
            "arrays": [_array_record(value) for value in row.arrays],
            "input_sha256": row.input_sha256, "status": row.status, "reason": row.reason,
            "effective_mass": cost_bits(row.effective_mass),
            "fusion_weight": cost_bits(row.fusion_weight), **_score_record(row.scores)}


def _cell_record(row: CellDiagnostic) -> dict:
    return {"cell": list(row.cell), "observation_indices": list(row.observation_indices),
            "input_sha256": row.input_sha256, "status": row.status, "reason": row.reason,
            "weight_mass": cost_bits(row.weight_mass), **_score_record(row.scores)}


def _result_record(result: DiagnosticResult) -> dict:
    return {"schema": RESULT_SCHEMA, "version": 1, "status": result.status,
            "reason": result.reason, "completeness": result.completeness,
            "state_ids": list(result.state_ids), "scorer": result.scorer.to_record(),
            "proposal": result.proposal.to_record(), "policy_sha256": result.policy_sha256,
            "universe_sha256": result.universe_sha256, "input_sha256": result.input_sha256,
            "reserved_work": asdict(result.reserved_work),
            "executed_work": asdict(result.executed_work),
            "reserved_payload_bytes": result.reserved_payload_bytes,
            "observations": [_observation_record(row) for row in result.observations],
            "cells": [_cell_record(row) for row in result.cells],
            "payload_sha256": result.payload_sha256}


def _vector_reservation(count: int) -> int:
    vector = 19 * count + 1  # 16 digits + quotes + comma; brackets replace last comma.
    ties = 1 + sum(len(str(index)) + 3 for index in range(count))
    # Replace null/null/[] (4+4+2) with complete vectors/worst singleton partition.
    return 2 * vector + ties - 10


def _cell_reservation(row_size: int, observation_sizes: tuple[int, ...]) -> int:
    """Pure arithmetic envelope, also usable to verify a dominated cap boundary."""
    size = row_size + sum(value + 1 for value in observation_sizes) + 2
    _limit(size, MAX_CELL_PAYLOAD_BYTES, "cell_payload_bytes")
    return size


def _reservation(policy, observations, metadata, state_ids, work, groups) -> tuple[int, Work]:
    reserved_work = replace(work, aggregate_components=len(groups) * len(state_ids))
    empty = DiagnosticResult(
        "complete", None, "exhaustive", state_ids, scorer=scorer_identity(),
        proposal=proposal_identity(), policy_sha256="0" * 64, universe_sha256="0" * 64,
        input_sha256="0" * 64, reserved_work=reserved_work, executed_work=reserved_work,
        reserved_payload_bytes=MAX_PAYLOAD_BYTES, payload_sha256="0" * 64,
    )
    size = _measure(_result_record(empty), 32_000_000)
    observation_sizes = []
    expansion = _vector_reservation(len(state_ids))
    for index, (observation, meta) in enumerate(zip(observations, metadata)):
        arrays = tuple(None if array is None else ArrayDigest(
            array.dtype.str, array.dtype.byteorder, array.shape, "0" * 64)
            for array in (observation.patch, observation.valid, observation.coverage))
        row = ObservationDiagnostic(index, meta, arrays, "0" * 64, "no_evidence",
                                    "zero_usable_coverage", 0.0, 0.0, None)
        observation_sizes.append(_measure(_observation_record(row)) + expansion)
    for cell, indices in groups:
        row = CellDiagnostic(cell, indices, "0" * 64, "no_evidence",
                             "no_usable_observations", 0.0, None)
        row_size = _measure(_cell_record(row)) + expansion
        # A cell envelope includes its observations as well as its aggregate row.
        _cell_reservation(row_size, tuple(observation_sizes[index] for index in indices))
        size += row_size + 1
    size += sum(row_size + 1 for row_size in observation_sizes)
    _limit(size, policy.max_proposal_payload_bytes, "payload_bytes")
    return size, reserved_work


def _validate_scores(scores: DiagnosticScores | None, count: int) -> None:
    if scores is None:
        return
    if (type(scores) is not DiagnosticScores or type(scores.absolute) is not tuple
            or len(scores.absolute) != count or scores != diagnostic_scores(scores.absolute)):
        raise _Refusal("score_contract")


def _refused(reason: str, cap: int, reserved: Work = Work(),
             executed: Work = Work()) -> DiagnosticResult:
    # No exclusion inventory, candidate table, partial rows or untrusted strings.
    # The fixed five-key refusal payload never carries the typed work counters.
    record = {"schema": RESULT_SCHEMA, "version": 1, "status": "refused",
              "reason": reason, "completeness": "refused"}
    size = _measure(record)
    payload = None if size > cap else _serialize(record)
    digest = None if payload is None else hashlib.sha256(payload).hexdigest()
    return DiagnosticResult("refused", reason, "refused", reserved_work=reserved,
                            executed_work=executed, payload=payload,
                            payload_sha256=digest)


def produce_diagnostics(policy: ImageDerivedOutputPolicyV2,
                        observations: tuple[FaceObservation, ...]) -> DiagnosticResult:
    """Return every candidate diagnostic, or one atomic typed refusal.

    Policy caps may lower the built-in resource profile. There is no injected
    scorer, callback, asset loading, production dispatch or state selection.
    """
    cap, reserved, executed = MAX_PAYLOAD_BYTES, Work(), Work()
    if type(policy) is ImageDerivedOutputPolicyV2:
        raw_cap = policy.max_proposal_payload_bytes
        if type(raw_cap) is int and 1 <= raw_cap <= MAX_PAYLOAD_BYTES:
            cap = raw_cap
    try:
        metadata, state_ids, reserved, descriptor_components = _admit(policy, observations)
        _policy_preflight(policy)
        replay = ImageDerivedOutputPolicyV2.from_record(policy.to_record())
        if replay != policy:
            raise _Refusal("policy_replay")
        universe = policy.universe
        if (universe.manifest.completeness.status != "exhaustive"
                or any(entry.disposition != "supported" for entry in universe.appearance_support)):
            raise _Refusal("incomplete_universe")
        if policy.scorer != scorer_identity() or policy.proposal != proposal_identity():
            raise _Refusal("builtin_identity")
        if state_ids != tuple(canonical_state_id(state) for state in universe.legal_states):
            raise _Refusal("state_identity")
        identities = set()
        for observation in observations:
            _validate_values(observation)
            if observation.identity in identities:
                raise _Refusal("duplicate_observation")
            identities.add(observation.identity)
        ordered = tuple(sorted(zip(observations, metadata), key=lambda pair: pair[0].identity))
        observations = tuple(pair[0] for pair in ordered)
        metadata = tuple(pair[1] for pair in ordered)
        grouped: dict[tuple[int, int, int], list[int]] = {}
        for index, observation in enumerate(observations):
            grouped.setdefault(observation.cell, []).append(index)
        groups = tuple((cell, tuple(indices)) for cell, indices in sorted(grouped.items()))
        payload_reservation, reserved = _reservation(
            policy, observations, metadata, state_ids, reserved, groups)
        scorer = AppearanceScorerV2(tuple(
            entry.appearance for entry in universe.appearance_support))
        rows = []
        for index, (observation, meta) in enumerate(zip(observations, metadata)):
            arrays = tuple(_hash_array(array) for array in (
                observation.patch, observation.valid, observation.coverage))
            input_digest = record_sha256({"metadata": asdict(meta),
                                          "arrays": [_array_record(array) for array in arrays]})
            observation_pixels = math.prod(observation.patch.shape[:2])
            observation_bytes = _observation_array_bytes(observation)
            # Hash unit: every array streamed and the complete observation digest exists.
            executed = replace(executed, hash_bytes=executed.hash_bytes + observation_bytes)
            features = observation_features(observation)
            scores = (None if features.reason is not None
                      else scorer.score(features, observation.slot))
            _validate_scores(scores, len(state_ids))
            # Observation unit: the whole row committed, scored or explicitly absent.
            executed = replace(
                executed, observations=executed.observations + 1,
                patch_pixels=executed.patch_pixels + observation_pixels,
                retained_array_bytes=executed.retained_array_bytes + observation_bytes,
                metadata_bytes=executed.metadata_bytes + _measure(
                    asdict(meta), MAX_OBSERVATION_METADATA_BYTES))
            if scores is not None:
                # Scoring unit: only a complete validated candidate vector counts.
                executed = replace(
                    executed, candidates=len(state_ids),
                    candidate_evaluations=executed.candidate_evaluations + len(state_ids),
                    candidate_pixel_work=(executed.candidate_pixel_work
                                          + len(state_ids) * observation_pixels),
                    descriptor_components=(executed.descriptor_components
                                           + descriptor_components))
            rows.append(ObservationDiagnostic(
                index, meta, arrays, input_digest,
                "no_evidence" if scores is None else "diagnostic", features.reason,
                features.mass, features.fusion_weight, scores))
        cells = []
        for cell, indices in groups:
            usable = tuple((rows[index].fusion_weight, rows[index].scores)
                           for index in indices if rows[index].scores is not None)
            scores = aggregate_scores(usable) if usable else None
            _validate_scores(scores, len(state_ids))
            mass = 0.0
            for weight, _ in usable:
                mass += weight
            if scores is not None:
                # Aggregate unit: only a complete validated cell vector counts.
                executed = replace(executed,
                                   aggregate_components=executed.aggregate_components
                                   + len(state_ids))
            cells.append(CellDiagnostic(
                cell, indices, record_sha256([rows[index].input_sha256 for index in indices]),
                "no_evidence" if scores is None else "diagnostic",
                "no_usable_observations" if scores is None else None, finite(mass), scores))
        result = DiagnosticResult(
            "complete", None, "exhaustive", state_ids, tuple(rows), tuple(cells),
            policy.scorer, policy.proposal, policy.policy_sha256, universe.universe_sha256,
            record_sha256([row.input_sha256 for row in rows]), reserved, executed,
            payload_reservation,
        )
        record = _result_record(result)
        size = _measure(record)
        if size > payload_reservation:
            raise _Refusal("reservation_contract")
        body = {key: value for key, value in record.items() if key != "payload_sha256"}
        digest = hashlib.sha256(_serialize(body)).hexdigest()
        result = replace(result, payload_sha256=digest)
        record = _result_record(result)
        size = _measure(record)
        if size > payload_reservation or size > cap:
            raise _Refusal("reservation_contract")
        payload = _serialize(record)
        if len(payload) != size:
            raise _Refusal("serialization_contract")
        return replace(result, payload=payload)
    except _Refusal as error:
        return _refused(error.reason, cap, reserved, executed)
    except (ValueError, TypeError, AttributeError, OverflowError, KeyError, UnicodeError):
        return _refused("invalid_contract", cap, reserved, executed)


def verify_payload(payload: bytes) -> dict:
    """Bounded exact binary64 diagnostic replay; rejects tamper and incomplete vectors."""
    if type(payload) is not bytes or len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("payload bytes exceed replay cap")
    record = json.loads(payload)
    if type(record) is not dict or _serialize(record) != payload:
        raise ValueError("noncanonical diagnostic payload")
    if record.get("schema") != RESULT_SCHEMA or record.get("version") != 1:
        raise ValueError("diagnostic schema mismatch")
    if record.get("status") == "refused":
        if set(record) != {"schema", "version", "status", "reason", "completeness"}:
            raise ValueError("invalid refusal payload")
        if record["completeness"] != "refused" or type(record["reason"]) is not str:
            raise ValueError("invalid refusal completeness")
        return record
    expected_fields = {
        "schema", "version", "status", "reason", "completeness", "state_ids", "scorer",
        "proposal", "policy_sha256", "universe_sha256", "input_sha256", "reserved_work",
        "executed_work", "reserved_payload_bytes", "observations", "cells", "payload_sha256",
    }
    if (set(record) != expected_fields or record["status"] != "complete"
            or record["completeness"] != "exhaustive" or record["reason"] is not None):
        raise ValueError("invalid complete diagnostic fields")
    reservation = record["reserved_payload_bytes"]
    if type(reservation) is not int or not len(payload) <= reservation <= MAX_PAYLOAD_BYTES:
        raise ValueError("invalid payload reservation")
    digest = record.get("payload_sha256")
    body = {key: value for key, value in record.items() if key != "payload_sha256"}
    if hashlib.sha256(_serialize(body)).hexdigest() != digest:
        raise ValueError("diagnostic digest mismatch")
    if (record["scorer"] != scorer_identity().to_record()
            or record["proposal"] != proposal_identity().to_record()):
        raise ValueError("diagnostic built-in identity mismatch")
    for label in ("reserved_work", "executed_work"):
        work = record[label]
        if (type(work) is not dict or set(work) != {item.name for item in fields(Work)}
                or any(type(value) is not int or value < 0 for value in work.values())):
            raise ValueError("invalid diagnostic work counters")
    if any(value > record["reserved_work"][label]
           for label, value in record["executed_work"].items()):
        raise ValueError("executed work exceeds reservation")
    states = record["state_ids"]
    if (type(states) is not list or not 1 <= len(states) <= MAX_CANDIDATES
            or any(type(state) is not str or len(state.encode("utf-8")) > MAX_STRING_BYTES
                   for state in states) or states != sorted(set(states))):
        raise ValueError("diagnostic universe mismatch")
    for row in (*record["observations"], *record["cells"]):
        if type(row) is not dict or row["status"] not in ("diagnostic", "no_evidence"):
            raise ValueError("invalid row status")
        if row["status"] == "no_evidence":
            if (row["absolute"] is not None or row["relative"] is not None or row["ties"] != []
                    or type(row["reason"]) is not str):
                raise ValueError("fabricated no-evidence scores")
        else:
            if row["reason"] is not None:
                raise ValueError("diagnostic row cannot carry a refusal reason")
            if len(row["absolute"]) != len(states) or len(row["relative"]) != len(states):
                raise ValueError("incomplete diagnostic vector")
            scores = diagnostic_scores(tuple(cost_from_bits(value) for value in row["absolute"]))
            if _score_record(scores) != {key: row[key] for key in ("absolute", "relative", "ties")}:
                raise ValueError("cost bits, relatives or ties do not replay")
    observations = record["observations"]
    if len(observations) > MAX_OBSERVATIONS:
        raise ValueError("diagnostic observation cap")
    keys = []
    for index, row in enumerate(observations):
        if row["index"] != index:
            raise ValueError("observation index mismatch")
        meta = row["metadata"]
        if set(meta) != {item.name for item in fields(ObservationMetadata)}:
            raise ValueError("invalid observation metadata fields")
        for label in ("support_frames", "provenance"):
            if meta[label] != sorted(set(meta[label])):
                raise ValueError("noncanonical observation references")
        cost_from_bits(row["effective_mass"])
        cost_from_bits(row["fusion_weight"])
        keys.append((tuple(meta["cell"]), meta["slot"], meta["frame_id"]))
        if record_sha256({"metadata": meta, "arrays": row["arrays"]}) != row["input_sha256"]:
            raise ValueError("observation digest mismatch")
    if keys != sorted(set(keys)):
        raise ValueError("noncanonical observation identities")
    cell_keys, indices = [], []
    for row in record["cells"]:
        cell = tuple(row["cell"])
        cell_keys.append(cell)
        refs = row["observation_indices"]
        if refs != sorted(set(refs)) or not refs:
            raise ValueError("invalid cell indices")
        for index in refs:
            if (type(index) is not int or not 0 <= index < len(observations)
                    or keys[index][0] != cell):
                raise ValueError("cross-cell observation index")
        cell_digest = record_sha256([observations[index]["input_sha256"] for index in refs])
        if cell_digest != row["input_sha256"]:
            raise ValueError("cell digest mismatch")
        usable = tuple((cost_from_bits(observations[index]["fusion_weight"]),
                        diagnostic_scores(tuple(cost_from_bits(value)
                                                for value in observations[index]["absolute"])))
                       for index in refs if observations[index]["status"] == "diagnostic")
        expected = aggregate_scores(usable) if usable else None
        if _score_record(expected) != {key: row[key] for key in ("absolute", "relative", "ties")}:
            raise ValueError("cell absolute aggregation does not replay")
        mass = 0.0
        for weight, _ in usable:
            mass += weight
        if cost_from_bits(row["weight_mass"]) != finite(mass):
            raise ValueError("cell weight mass mismatch")
        indices.extend(refs)
    if cell_keys != sorted(set(cell_keys)) or sorted(indices) != list(range(len(observations))):
        raise ValueError("incomplete cell partition")
    if record_sha256([row["input_sha256"] for row in observations]) != record["input_sha256"]:
        raise ValueError("input digest mismatch")
    return record
