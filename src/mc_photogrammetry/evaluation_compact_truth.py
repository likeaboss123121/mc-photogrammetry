"""Evaluation-only compact view of validated rendered truth evidence."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib, json
import numpy as np

from mc_photogrammetry import evaluation_frozen_planes as full

SCHEMA = "mc-photogrammetry-compact-rendered-truth-source-v1"

def _sha(data): return hashlib.sha256(data).hexdigest()
def _canonical(value): return json.dumps(value, sort_keys=True, separators=(",", ":"),
                                         allow_nan=False).encode()
def _array(value):
    return {"dtype": value.dtype.str, "shape": list(value.shape),
            "sha256": _sha(value.tobytes(order="C"))}

@dataclass(frozen=True, slots=True)
class CompactRenderedTruthSourceV1:
    schema: str
    source_evidence_sha256: str
    canonical_source_record: bytes
    camera: object
    truth_sha256: str
    vanilla_jar_sha256: str
    palette_sha256: str
    rgb: np.ndarray
    ownership: np.ndarray
    usable: np.ndarray
    owner_index: tuple
    surface_descriptors: tuple
    view_sha256: str = ""

    def __post_init__(self):
        if (self.schema != SCHEMA or len(self.source_evidence_sha256) != 64
                or not isinstance(self.canonical_source_record, bytes)):
            raise ValueError("compact truth source is malformed")
        try:
            record = json.loads(self.canonical_source_record)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("compact source record is malformed") from error
        source_payload=dict(record); source_digest=source_payload.pop("evidence_sha256",None)
        if (_canonical(record) != self.canonical_source_record
                or record.get("schema") != "mc-photogrammetry-rendered-truth-evidence-v1"
                or source_digest != self.source_evidence_sha256
                or _sha(_canonical(source_payload)) != self.source_evidence_sha256):
            raise ValueError("compact source record is not canonical or bound")
        arrays=(self.rgb,self.ownership,self.usable)
        if (any(not isinstance(x,np.ndarray) or x.flags.writeable for x in arrays)
                or self.rgb.dtype != np.uint8 or self.rgb.ndim != 3
                or self.rgb.shape[2] != 3 or self.ownership.dtype.kind not in "iu"
                or self.ownership.shape != self.rgb.shape[:2]
                or self.usable.dtype != np.bool_ or self.usable.shape != self.rgb.shape[:2]):
            raise ValueError("compact truth arrays are mutable or malformed")
        expected_surfaces=tuple((x["direction"],x["coordinate"],x["mask"]["dtype"],
                                 tuple(x["mask"]["shape"]),x["mask"]["sha256"])
                                for x in record["surfaces"])
        if (record["camera"] != self.camera.to_dict()
                or record["truth_sha256"] != self.truth_sha256
                or record["vanilla_jar_sha256"] != self.vanilla_jar_sha256
                or record["palette_sha256"] != self.palette_sha256
                or record["rgb"] != _array(self.rgb)
                or record["ownership"] != _array(self.ownership)
                or record["usable"] != _array(self.usable)
                or tuple((tuple(x[0]),x[1]) for x in record["owner_index"]) != self.owner_index
                or expected_surfaces != self.surface_descriptors):
            raise ValueError("compact truth source does not bind canonical evidence")
        digest=_sha(_canonical(self.to_record(False)))
        if self.view_sha256 not in ("",digest): raise ValueError("compact view digest mismatch")
        object.__setattr__(self,"view_sha256",digest)

    @property
    def evidence_sha256(self): return self.source_evidence_sha256

    def to_record(self, include_digest=True):
        result={"schema":self.schema,"source_evidence_sha256":self.source_evidence_sha256,
                "canonical_source_sha256":_sha(self.canonical_source_record),
                "camera":self.camera.to_dict(),"truth_sha256":self.truth_sha256,
                "vanilla_jar_sha256":self.vanilla_jar_sha256,
                "palette_sha256":self.palette_sha256,"rgb":_array(self.rgb),
                "ownership":_array(self.ownership),"usable":_array(self.usable),
                "owner_index":[[list(p),s] for p,s in self.owner_index],
                "surface_descriptors":[[d,c,t,list(sh),h] for d,c,t,sh,h in self.surface_descriptors]}
        if include_digest: result["view_sha256"]=self.view_sha256
        return result

def compact_rendered_truth(evidence):
    evidence=full.RenderedTruthEvidence(**{n:getattr(evidence,n)
                                           for n in evidence.__dataclass_fields__})
    record=evidence.to_record(); canonical=_canonical(record)
    surfaces=tuple((s.direction,s.coordinate,s.mask.dtype.str,s.mask.shape,
                    _sha(s.mask.tobytes(order="C"))) for s in evidence.surfaces)
    return CompactRenderedTruthSourceV1(
        SCHEMA,evidence.evidence_sha256,canonical,evidence.camera,evidence.truth_sha256,
        evidence.vanilla_jar_sha256,evidence.palette_sha256,evidence.rgb,
        evidence.ownership,evidence.usable,evidence.owner_index,surfaces)

def replay_compact_signature(*, signature_evidence, truth_source=None,
                             truth_evidence=None,
                             witness_budget=None, witness_payload_sink=None):
    """Exact sampling replay against the compact RGB/usable view."""
    from mc_photogrammetry import evaluation_signature_sampling as sampling
    from mc_photogrammetry.pipeline import image_derived_signatures as signatures
    from mc_photogrammetry.reconstruction import lattice_planes
    from mc_photogrammetry.vision.gui_mask import create_gui_mask
    signature_sha=getattr(signature_evidence,"signature_sha256","")
    supplied_truth = truth_source if truth_source is not None else truth_evidence
    truth_sha=getattr(supplied_truth,"evidence_sha256","")
    if witness_payload_sink is not None and len(witness_payload_sink):
        raise ValueError("signature witness payload sink must be empty")
    try:
        if truth_source is not None and truth_evidence is not None:
            raise ValueError("compact replay accepts one truth source")
        signature_evidence=signatures.SignatureEvidence(**{
            n:getattr(signature_evidence,n)
            for n in signature_evidence.__dataclass_fields__})
        truth_source=CompactRenderedTruthSourceV1(**{
            n:getattr(supplied_truth,n) for n in supplied_truth.__dataclass_fields__})
        expected=np.logical_not(create_gui_mask(*truth_source.usable.shape))
        if not np.array_equal(expected,truth_source.usable):
            return sampling._refusal("usable_mask_mismatch","usable_mask",
                "truth usable mask differs from the production GUI mask",signature_sha,
                truth_source.evidence_sha256)
        sink=[]
        replay=lattice_planes.plane_signature(
            truth_source.rgb,signature_evidence.signature.reading,
            signature_evidence.blocks_per_period,usable=truth_source.usable,
            witness_sink=sink,witness_budget=witness_budget)
        if len(sink)!=1:
            return sampling._refusal("replay_mismatch","witness",
                "sampling witness publication is incomplete",signature_sha,
                truth_source.evidence_sha256)
        from mc_photogrammetry.reconstruction import lattice
        if isinstance(sink[0],lattice.WitnessRefusalV1):
            x=sink[0]; return sampling._refusal("budget_exhausted",x.stage,
                f"limit={x.limit};planned={x.planned};actual={x.actual}",signature_sha,
                truth_source.evidence_sha256)
        accepted=signature_evidence.signature
        if not (replay.reading==accepted.reading and replay.blocks_per_period==accepted.blocks_per_period
                and replay.origin_uv==accepted.origin_uv and replay.region_footprint==accepted.region_footprint
                and np.array_equal(replay.cells,accepted.cells)
                and np.array_equal(replay.interior,accepted.interior)
                and np.array_equal(replay.coverage,accepted.coverage)):
            return sampling._refusal("replay_mismatch","signature",
                "replayed signature differs from accepted evidence",signature_sha,
                truth_source.evidence_sha256)
        binding=sampling.SignatureSamplingBindingV1(sampling.SCHEMA,
            signature_evidence.frame_id,signature_evidence.canonical_index,
            signature_evidence.alternative_index,signature_sha,
            truth_source.evidence_sha256,sink[0].witness_sha256)
        if witness_payload_sink is not None:
            witness_payload_sink.append(sampling.SignatureSamplingWitnessPayloadV1(binding,sink[0]))
        return binding
    except (AttributeError,TypeError,ValueError,OverflowError) as error:
        return sampling._refusal("provenance_mismatch","input",type(error).__name__,
                                 signature_sha,truth_sha)
