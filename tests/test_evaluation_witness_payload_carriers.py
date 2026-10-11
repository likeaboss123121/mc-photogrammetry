from dataclasses import replace

import numpy as np
import pytest

from mc_photogrammetry import evaluation_frozen_planes as truth_mod
from mc_photogrammetry import evaluation_signature_sampling as sampling_mod
from mc_photogrammetry import synthetic_targeted_capture_preflight as preflight
from mc_photogrammetry.reconstruction import plane_graph
from test_evaluation_signature_sampling import _fixture
from test_synthetic_preflight_evidence_bundle import _rows


def _matcher_payload(rows):
    _evidences, _frames, _owners, signatures, envelopes, _trace = rows
    pairs = np.asarray([[0, 0, 0, 0]], dtype=np.int64)
    pairs.flags.writeable = False
    homography = np.eye(3)
    homography.flags.writeable = False
    witness = plane_graph.MatcherPairWitnessV1(
        "mc-photogrammetry-matcher-pair-witness-v1",
        plane_graph.PLANE_CORRELATION_ALGORITHM, 32, 0, homography,
        "1" * 64, (132, 132), (0, 0), 1, pairs)
    payload = plane_graph.GraphMatcherWitnessPayloadV1(envelopes[0], witness)
    assert payload.envelope.left_signature_sha256 == signatures[0].signature_sha256
    return payload


def test_signature_payload_revalidates_exact_nested_witness_and_is_deterministic():
    truth, signature = _fixture()
    payloads = []
    binding = sampling_mod.replay_signature_sampling_witness(
        signature_evidence=signature, truth_evidence=truth,
        witness_payload_sink=payloads)
    assert len(payloads) == 1
    payload = payloads[0]
    assert isinstance(payload, sampling_mod.SignatureSamplingWitnessPayloadV1)
    assert payload.binding == binding
    assert sampling_mod.SignatureSamplingWitnessPayloadV1(
        payload.binding, payload.witness) == payload
    with pytest.raises(ValueError, match="digest mismatch"):
        sampling_mod.SignatureSamplingWitnessPayloadV1(
            replace(payload.binding, witness_sha256="0" * 64,
                    report_sha256=""), payload.witness)
    with pytest.raises(ValueError, match="must be empty"):
        sampling_mod.replay_signature_sampling_witness(
            signature_evidence=signature, truth_evidence=truth,
            witness_payload_sink=[payload])


def test_matcher_payload_revalidates_shift_support_and_immutable_arrays():
    payload = _matcher_payload(_rows())
    assert plane_graph.GraphMatcherWitnessPayloadV1(
        payload.envelope, payload.witness) == payload
    with pytest.raises(ValueError, match="does not bind"):
        plane_graph.GraphMatcherWitnessPayloadV1(
            replace(payload.envelope, overlap_cells=2, envelope_sha256=""),
            payload.witness)
    mutable = payload.witness.logical_pairs.copy()
    with pytest.raises(ValueError, match="mutable"):
        replace(payload.witness, logical_pairs=mutable, witness_sha256="")


def test_private_bundle_joins_by_signature_hash_and_refuses_order_and_cap():
    rows = _rows()
    evidences, frames, owner_bindings, signature_bindings, envelopes, trace = rows
    public = preflight._assemble_evidence_bundle(*rows)
    # Synthetic owner witnesses cannot be forged from hashes: use explicit wrong
    # types to verify the private boundary fails closed before publication.
    missing = preflight._assemble_witness_payload(
        public, evidences, (), (), (), preflight.MAX_WITNESS_PAYLOAD_BYTES)
    assert missing.status == "refused" and missing.stage == "witness_payload"
    capped = preflight._assemble_witness_payload(
        public, evidences, (), (), (), 0)
    assert capped.status == "refused"
    matcher = _matcher_payload(rows)
    # Endpoint joins are signature hashes. Plane indices are deliberately not
    # compared with global canonical reading indices.
    assert matcher.envelope.left_signature_sha256 == signature_bindings[0].signature_sha256
    assert matcher.envelope.right_signature_sha256 == signature_bindings[1].signature_sha256
    assert matcher.envelope.match_sha256 not in {
        matcher.envelope.envelope_sha256, matcher.witness.witness_sha256}
    assert preflight._payload_nbytes(matcher) > matcher.witness.logical_pairs.nbytes


def test_payload_carriers_reject_wrong_nested_schema():
    payload = _matcher_payload(_rows())
    with pytest.raises(ValueError):
        replace(payload.envelope, schema="unknown", envelope_sha256="")
    truth, signature = _fixture()
    sink = []
    sampling_mod.replay_signature_sampling_witness(
        signature_evidence=signature, truth_evidence=truth,
        witness_payload_sink=sink)
    with pytest.raises(ValueError):
        replace(sink[0].witness, schema="unknown", witness_sha256="")


def test_aggregate_payload_replays_nested_carriers():
    # Constructor must touch nested carrier constructors rather than accepting
    # only their outward digest strings.
    class Foreign:
        pass
    rows = _rows()
    public = preflight._assemble_evidence_bundle(*rows)
    with pytest.raises((AttributeError, TypeError, ValueError)):
        preflight.SyntheticPreflightWitnessPayloadV1(
            public, (Foreign(),), (), ())
