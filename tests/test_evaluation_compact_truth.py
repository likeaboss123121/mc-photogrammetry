from dataclasses import replace
import json
import numpy as np
import pytest

from mc_photogrammetry import evaluation_compact_truth as compact
from mc_photogrammetry import evaluation_signature_sampling as sampling
from mc_photogrammetry import evaluation_frozen_planes as frozen
from mc_photogrammetry.export import attribution
from test_evaluation_signature_sampling import _fixture as sampling_fixture
from mc_photogrammetry import evaluation_surface_support_streaming as streaming
from test_evaluation_surface_support import _payload_fixture

MERGE="9"*64

def test_compact_source_discards_masks_and_binds_canonical_view():
    payload,left,*_= _payload_fixture()
    view=compact.compact_rendered_truth(left)
    assert view.source_evidence_sha256==left.evidence_sha256
    assert not hasattr(view,"surfaces")
    assert not any(value is surface.mask for value in view.__dataclass_fields__.values()
                   for surface in left.surfaces)
    assert not view.rgb.flags.writeable and not view.ownership.flags.writeable
    assert compact.CompactRenderedTruthSourceV1(**{
        n:getattr(view,n) for n in view.__dataclass_fields__})==view
    with pytest.raises(ValueError,match="digest mismatch"):
        replace(view,view_sha256="0"*64)

def test_full_and_compact_summary_and_sampling_are_identical():
    payload,left,right,ls,rs=_payload_fixture()
    view=compact.compact_rendered_truth(left)
    full=streaming.summarize_signature_ownership(
        bi_merge_sha256=MERGE,signature_evidence=ls,
        sampling_payload=payload.signatures[0],truth_evidence=left,
        ownership_payload=payload.ownership[0])
    reduced=streaming.summarize_signature_ownership(
        bi_merge_sha256=MERGE,signature_evidence=ls,
        sampling_payload=payload.signatures[0],truth_evidence=view,
        ownership_payload=payload.ownership[0])
    assert reduced==full
    full_sink=[]
    full_binding=sampling.replay_signature_sampling_witness(
        signature_evidence=ls,truth_evidence=left,
        witness_payload_sink=full_sink)
    sink=[]
    binding=compact.replay_compact_signature(
        signature_evidence=ls,truth_source=view,witness_payload_sink=sink)
    assert binding==full_binding
    assert sink==full_sink

def test_compact_tamper_mutability_and_source_record_refuse():
    _payload,left,*_= _payload_fixture(); view=compact.compact_rendered_truth(left)
    rgb=view.rgb.copy()
    with pytest.raises(ValueError,match="mutable"):
        replace(view,rgb=rgb,view_sha256="")
    frozen=rgb.copy(); frozen.reshape(-1)[0]^=1; frozen.flags.writeable=False
    with pytest.raises(ValueError,match="does not bind"):
        replace(view,rgb=frozen,view_sha256="")
    with pytest.raises(ValueError,match="canonical or bound"):
        replace(view,canonical_source_record=view.canonical_source_record+b" ",
                view_sha256="")

def test_compaction_does_not_mutate_public_graph_or_source_arrays():
    _payload,left,*_= _payload_fixture()
    before=left.evidence_sha256; rgb=left.rgb.tobytes(); masks=tuple(
        surface.mask.tobytes() for surface in left.surfaces)
    view=compact.compact_rendered_truth(left)
    assert left.evidence_sha256==before and left.rgb.tobytes()==rgb
    assert tuple(surface.mask.tobytes() for surface in left.surfaces)==masks
    assert view.source_evidence_sha256==before

def test_real_coordinator_replays_compact_binding_and_witness():
    truth,signature=sampling_fixture(); view=compact.compact_rendered_truth(truth)
    row=attribution.RendererOwnerRowV1(
        0,(0,0,0),"state","block",0,"east",(),(1.,0.,0.),
        "a"*64,"",(0,1,1,1))
    witness=attribution.RendererOwnershipWitnessV1(
        "mc-photogrammetry-renderer-ownership-witness-v1",1.,.5,
        "strict-nearer-inverse-depth-v1","face-order-stable-v1",(row,))
    binding=frozen.bind_renderer_ownership_witness(truth,witness)
    owner=frozen.RenderedTruthOwnershipWitnessPayloadV1(binding,witness)
    sink=[]
    result=streaming.summarize_endpoints_atomically((dict(
        bi_merge_sha256=MERGE,signature_evidence=signature,
        truth_evidence=view,ownership_payload=owner,
        replay=compact.replay_compact_signature),),sink)
    assert len(result)==len(sink)==1
    assert result[0].truth_evidence_sha256==truth.evidence_sha256
    assert result[0].sampling_witness_sha256

def test_canonical_source_digest_and_schema_are_recomputed():
    _payload,left,*_= _payload_fixture(); view=compact.compact_rendered_truth(left)
    record=json.loads(view.canonical_source_record)
    record["truth_sha256"]="0"*64
    forged=json.dumps(record,sort_keys=True,separators=(",",":")).encode()
    with pytest.raises(ValueError,match="canonical or bound"):
        replace(view,canonical_source_record=forged,view_sha256="")

def test_compact_replay_stale_and_malformed_signature_are_typed_refusals():
    truth,signature=sampling_fixture(); view=compact.compact_rendered_truth(truth)
    stale=object.__new__(type(signature))
    for name in signature.__dataclass_fields__:
        object.__setattr__(stale,name,getattr(signature,name))
    object.__setattr__(stale,"signature_sha256","0"*64)
    refusal=compact.replay_compact_signature(
        signature_evidence=stale,truth_source=view)
    assert refusal.status=="refused" and refusal.reason=="provenance_mismatch"
    assert refusal.stage=="input" and refusal.signature_sha256=="0"*64
    malformed=compact.replay_compact_signature(
        signature_evidence=object(),truth_source=view)
    assert malformed.status=="refused" and malformed.reason=="provenance_mismatch"
    assert malformed.stage=="input"
    record=json.loads(view.canonical_source_record); record["schema"]="foreign"
    forged=json.dumps(record,sort_keys=True,separators=(",",":")).encode()
    with pytest.raises(ValueError,match="canonical or bound"):
        replace(view,canonical_source_record=forged,view_sha256="")
