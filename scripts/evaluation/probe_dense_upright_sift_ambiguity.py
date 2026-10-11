"""Fixed-source dense upright SIFT experiment; no physical association.

Imports do not access pixels or invoke detector/descriptor/matcher services.
Pure synthetic descriptor tables exercise the arithmetic without image runtime.
The separate absent lead authorization gates all actual source acquisition.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from fractions import Fraction
import base64
import binascii
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PROBE = "scripts/evaluation/probe_dense_upright_sift_ambiguity.py"
TEST = "tests/test_dense_upright_sift_ambiguity.py"
AUTH = "agent swarm instantiation template/runs/RESULT-LEAD-DENSE-UPRIGHT-SIFT-AMBIGUITY-AUTH-02-2026-10-09.json"
OUTPUT = "exports/diagnostics/dense-upright-sift-ambiguity-2026-10-09-repair-01"
SCHEMA = "dense-upright-sift-ambiguity-v2"
INVENTORY = "dense-upright-sift-prescribed-native-grid-v2"
TABLE_SCHEMA = "dense-upright-descriptor-table-v2"
STATIC_METADATA_BYTES = 262144
LOG_BYTES = 65536
COMPLETION_BYTES = 65536
TREATMENT = {"spacing":16,"offset":8,"size":16.0,"angle":0.0,"ratio":0.75,
             "preprocessing":"correspondence._to_grey RGB2GRAY; unmodified raster; no usable-mask zeroing",
             "usable_center":"production usable_mask_for; no erosion","mask_boundary_pad_px":0.0}
LIMITS = {"frames":2,"decoded_pixels":2880000,"grid_centers":11200,
          "mask_memberships":16800,"left_descriptor_sites":256,
          "right_descriptor_sites":4096,"total_descriptor_sites":4352,
          "cross_frame_comparisons":1048576,"common_witness_nodes":4096,
          "common_witness_determinants":32768,
          "wall_seconds":180,"job_committed_bytes":2147483648,
          "evidence_bytes":8388608,"artifact_bytes":20971520}
CAPTURE = 'exports/datasets/fresh-c4-canal-town-framed-2026-09-12/scenes/canal-town-architectural-component-bbox/captures/'
DIAGNOSTIC = 'exports/diagnostics/endpoint-conflict-mask-bundle-2026-10-09/diagnostic.json'
DIAGNOSTIC_SHA = 'a4bb71e275b277a8b52a8b14884cb72a8be71df698a5f9e4c0d44894daf3aa98'
SOURCE_PINS = {'scripts/evaluation/image_point_candidate_roster.py': '0e9a027e1209f3a96c9e2bd7443311a14e654f7bd3440f30cab0404f32f28ca2', 'scripts/evaluation/diagnose_image_geometry_failure.py': '5c3bdb9ed5dc34b021f3e60c92d0b7cd5ffa2aba8aea35225354213107d0d2af', 'src/mc_photogrammetry/input/image.py': '43c318a45bc6965b6a9d3d8f43fc2375052cbe5ad2b13b07a7b7b4313f0bf732', 'src/mc_photogrammetry/pipeline/correspondence.py': '192a4851e428d8116e0fbffd28c7ff157d1aedd8035e499c6df7ec2a9d787fa1', 'src/mc_photogrammetry/pipeline/image_derived_session.py': 'e48414bf9eea6a7da7a7cc7d8c19865e0494953ac8796d9f3cd2e57e81b6248e', 'src/mc_photogrammetry/vision/gui_mask.py': '1d8a6d9ed9537a2e5dd98c794aa9c45b07c748090e1a17f9db3d230e5e74cda9', 'src/mc_photogrammetry/reconstruction/lattice_planes.py': 'c416bea087eb1c9228fa0b17f78fb8177b49e711612e4322af989aa15e65995b', 'src/mc_photogrammetry/pipeline/image_derived_signatures.py': '7e287aecb6c1e6bb111c5d5a6c33abc876b6083ce48abdf3c4fff1306819e43f', 'src/mc_photogrammetry/pipeline/image_derived_lattice_measure.py': 'fb061bbfaa14bfb969e4349f1a01c080ee338a558d3cd6521425eb1d8231ed8e', 'src/mc_photogrammetry/pipeline/image_derived_endpoint_selection.py': 'f8065946eaedb95cdbb8d2bfceace0246bf1369d2f8683f2fd555aa1904a7c0a', 'src/mc_photogrammetry/pipeline/image_derived_joint_graph.py': '60f4f55bc887bef50c60427a00fca54f5ad854a1785ebd5b1319c4b823df25ab', 'scripts/evaluation/acquire_endpoint_conflict_native_roster.py': '6cb854e1800779514efe26e6ed3ea83a5c7a0975743380e8969d9bc46c6cc14c', 'scripts/evaluation/probe_native_feature_density.py': 'ff2a8f647362c2cc101706fe0a660732681d1792f37d13848360b3bc4e84a467'}
MASK_PINS = ((('image-0000', 2), '338a47a2cb081a1991014fd59b9413f04e2769195f39c373335e31dec9d25098', '3ecfb4b4709962ecbc5a8e122858c54a5c3334e7a0451ea55ce02bcb0381019a', 'bd86f23c0191e91a9c93220c50e798fabd5e3e43c576a5fd7cd5b8bd6e94d708'), (('image-0000', 3), '61b803612ab055368b8bdab037441f780b9014eace8c117b41da6b28cd77d55d', '85b28d1243dd715b2bddb02b129bd24df4e868129541b1b71d29f68e0354367c', 'cc7f4bbdca723ea630ddd7a2d3abfb3b97632ee8cc1569ac6465b7b8aaeb989e'), (('image-0001', 1), 'd28eb153b89b3850b00baa218a90584ff92fa4b036048551362fdff50c832d33', '1e275ace2937394b743e1af64b16c5e767a579f00997123d2d034a63f5211977', 'e557cba4a4a0401e6a2ebd5f7146ecee61d1727348c5e68310db7ac49e32aaf9'))
INPUTS = ((CAPTURE+"view-canal-town-architectural-component-bbox-05.png","6f844d5e3600e853bf3045fc25b55a271c0b665226053602b6d30c945f0f016b"), (CAPTURE+"view-canal-town-architectural-component-bbox-06.png","53c38526d378a0597ab6e00fb5e01a6c7aa0e2dc11619c1b7eaa3cfc0ccf8923"))


class Refusal(RuntimeError):
    """Terminal resource/integrity refusal, never a negative experimental finding."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise Refusal("duplicate_json_key")
        result[key] = value
    return result


def parse(data):
    return json.loads(data, object_pairs_hook=_unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(Refusal("nonfinite_json")))


def read(path, limit):
    with Path(path).open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise Refusal("source_byte_limit")
    return data


def pinned(path, pin, limit=4194304):
    data = read(path, limit)
    if type(pin) is not str or sha(data) != pin.lower():
        raise Refusal("source_pin_mismatch: " + str(path))
    return data


def frame_site_cap(frame_index):
    if frame_index not in (0,1):
        raise Refusal('invalid_frame_index')
    return LIMITS['left_descriptor_sites'] if frame_index==0 else LIMITS['right_descriptor_sites']


def encode_table(descriptors, frame_index):
    """Lossless C-order little-endian float32; cap checks precede byte allocation."""
    np=_np()
    if (not isinstance(descriptors,np.ndarray) or descriptors.dtype!=np.float32
            or descriptors.ndim!=2 or descriptors.shape[1]!=128):
        raise Refusal('descriptor_table_shape_dtype')
    rows=len(descriptors)
    if rows>frame_site_cap(frame_index):
        raise Refusal('descriptor_table_row_cap')
    if not np.isfinite(descriptors).all():
        raise Refusal('nonfinite_descriptor_payload')
    data=descriptors.astype('<f4',copy=False).tobytes(order='C')
    return {'schema':TABLE_SCHEMA,'dtype':'<f4','encoding':'base64-rfc4648-canonical',
            'order':'C; classes ordered by(y,x)','width':128,'row_count':rows,
            'byte_count':len(data),'content_sha256':sha(data),
            'data_base64':base64.b64encode(data).decode('ascii')}


def decode_table(record, frame_index, classes=None):
    """Refuse overbound header/text/row refs before decoding any payload bytes."""
    keys={'schema','dtype','encoding','order','width','row_count','byte_count','content_sha256','data_base64'}
    if type(record) is not dict or set(record)!=keys:
        raise Refusal('descriptor_table_fields')
    rows=record['row_count'];count=record['byte_count'];encoded=record['data_base64']
    if (type(rows) is not int or not 0<=rows<=frame_site_cap(frame_index)
            or type(count) is not int or count!=rows*128*4
            or record['schema']!=TABLE_SCHEMA or record['dtype']!='<f4' or record['width']!=128
            or record['encoding']!='base64-rfc4648-canonical' or record['order']!='C; classes ordered by(y,x)'
            or type(encoded) is not str or len(encoded)!=4*((count+2)//3)
            or type(record['content_sha256']) is not str or len(record['content_sha256'])!=64):
        raise Refusal('descriptor_table_bound_binding')
    if classes is not None:
        if len(classes)!=rows:
            raise Refusal('descriptor_table_reference_count')
        refs=[feature['content']['descriptor_row'] for feature in classes]
        if (any(type(value) is not int or not 0<=value<rows for value in refs)
                or refs!=list(range(rows)) or len({tuple(f['xy']) for f in classes})!=rows
                or [f['xy'] for f in classes]!=sorted([f['xy'] for f in classes],key=lambda p:(p[1],p[0]))):
            raise Refusal('descriptor_table_row_references')
    try:
        raw=base64.b64decode(encoded,validate=True)
    except (binascii.Error,ValueError,UnicodeEncodeError) as error:
        raise Refusal('invalid_descriptor_base64') from error
    if (len(raw)!=count or base64.b64encode(raw).decode('ascii')!=encoded
            or sha(raw)!=record['content_sha256']):
        raise Refusal('descriptor_table_canonical_digest')
    np=_np();table=np.frombuffer(raw,dtype='<f4').reshape(rows,128)
    if not np.isfinite(table).all():
        raise Refusal('nonfinite_descriptor_payload')
    return table


def _maximum_shape(left_rows, right_rows, group_rows):
    """A byte ceiling, intentionally not a valid observation or fitted model.

    Every dynamic ID is64 unescaped ASCII; all binary64 JSON numbers are at
    most24 characters; every bounded integer fits8 digits. Repeated containers
    model the largest schema shapes without any combination/distance work.
    """
    identifier='f'*64;number=-1.7976931348623157e308;xy=[number,number];integer=99999999
    work={key:integer for key in ('requested','charged','processed','uncharged','unprocessed')}
    envelope={'schema':'dense-upright-size-envelope-v2','frame_site_limits':[256,4096],
        'metadata_bytes':integer,'metadata_cap':STATIC_METADATA_BYTES,'evidence_upper_bound':integer,
        'completion_upper_bound':COMPLETION_BYTES,'log_cap':LOG_BYTES,'aggregate_upper_bound':integer,
        'evidence_cap':LIMITS['evidence_bytes'],'artifact_cap':LIMITS['artifact_bytes'],
        'method':'canonical maximum-shaped record + three bounded static-metadata expansions'}
    counts={key:integer for key in ('planned_sites','returned_rows','missing_sites','duplicate_rows',
        'coincident_conflicting_sites','coincident_conflicting_rows','zero_rows','nonfinite_rows',
        'returned_duplicate_coordinates','valid_distinct_sites','distinct_coordinates')}
    counts.update(returned_descriptor_table_sha256=identifier,returned_keypoints_sha256=identifier)
    def inventory_shape(rows):
        features=[{'feature_id':identifier,'xy':xy,'multiplicity':9999,
            'content':{'xy':xy,'size':16.0,'angle':0.0,'descriptor_row':9999}} for _ in range(rows)]
        return {'anchor':{'schema':INVENTORY,'treatment':TREATMENT,'frame_source':{},
            'site_plan_sha256':identifier,'runtime':{},'descriptor_table_sha256':identifier},
            'anchor_sha256':identifier,'counts':counts,'classes':features,
            'descriptor_table':{'schema':TABLE_SCHEMA,'dtype':'<f4','encoding':'base64-rfc4648-canonical',
                'order':'C; classes ordered by(y,x)','width':128,'row_count':9999,'byte_count':integer,
                'content_sha256':identifier,'data_base64':'A'*(4*((rows*512+2)//3))}}
    def supply_shape(rows):
        return {'distinct_sites':9999,'distinct_coordinates':9999,'bins':[[identifier]*rows]+[[] for _ in range(15)],
            'occupied_bins':99,'all_eligible_general_position':False,'unchanged_first_four_ids':[identifier]*4,
            'unchanged_first_four_ready':False}
    def group_shape(rows):
        pair={'left_id':identifier,'right_id':identifier,'left_xy':xy,'right_xy':xy,
              'distance_squared':number,'pair_id':identifier}
        coordinates={key:pair[key] for key in ('pair_id','left_id','right_id','left_xy','right_xy')}
        nearest={'origin':9999,'nearest':9999,'second':9999,'nearest_squared':number,
                 'second_squared':number,'status':'absent_second_neighbor'}
        direction={key:9999 for key in ('origins','absent_second_neighbor','nearest_distance_tie','ratio_rejected','ratio_accepted')}
        return {'source_row':999,'supply':{'left_eligible_sites':9999,'right_eligible_sites':9999,
            'left':supply_shape(rows),'right':supply_shape(right_rows)},
            'discrimination':{'forward':direction,'reverse':direction,'forward_evidence':[nearest]*rows,
                'reverse_evidence':[nearest]*right_rows,'unambiguous_reciprocal_pairs':9999,'pairs':[pair]*rows,
                'distinct_left_endpoints':9999,'distinct_right_endpoints':9999,
                'left_endpoint_general_position':False,'right_endpoint_general_position':False,
                'common_four_witness':{'status':'exhaustive_no_witness','work_nodes':9999,'determinants':99999,
                    'pair_ids':[identifier]*4,'coordinates':[coordinates]*4,'source_binding_sha256':identifier}},
            'status':'reciprocal_candidates_without_common_nondegenerate_witness','physical_association':'unknown'}
    return {'schema':SCHEMA,'status':'reciprocal_candidates_without_common_nondegenerate_witness',
        'treatment':TREATMENT,'limits':LIMITS,'sources':{},'plan':{},
        'inventories':[inventory_shape(left_rows),inventory_shape(right_rows)],
        'groups':[group_shape(rows) for rows in group_rows],'comparison_work':work,
        'common_witness_work':work,'common_witness_determinant_work':work,'physical_association':'unknown',
        'interpretation':'Supply/discrimination only; no accepted association or accuracy.',
        'size_bound':envelope,'record_sha256':identifier}


def record_size_ceiling(left_rows=256, right_rows=4096, group_rows=(256,256)):
    if (type(left_rows) is not int or type(right_rows) is not int or not 0<=left_rows<=256
            or not 0<=right_rows<=4096 or len(group_rows)!=2
            or any(type(value) is not int or not 0<=value<=left_rows for value in group_rows)):
        raise Refusal('size_model_count_cap')
    dynamic=len(canonical(_maximum_shape(left_rows,right_rows,group_rows)))+1
    evidence=dynamic+3*STATIC_METADATA_BYTES
    aggregate=max(2*evidence+LOG_BYTES,evidence+2*COMPLETION_BYTES+LOG_BYTES)
    return {'dynamic_canonical_bytes':dynamic,'evidence_upper_bound':evidence,'aggregate_upper_bound':aggregate}


def prepare_size_bound(plan, bindings, runtime, *, evidence_cap=None, artifact_cap=None):
    metadata=len(canonical({'plan':plan,'sources':bindings,'runtime':runtime}))
    if metadata>STATIC_METADATA_BYTES:
        raise Refusal('static_metadata_bound_before_compute')
    completion_shape={'schema':'dense-upright-sift-ambiguity-completion-v2',
        'status':'reciprocal_candidates_without_common_nondegenerate_witness','ambiguity_sha256':'f'*64,
        'ambiguity_bytes':99999999,'sources':bindings,'limits':LIMITS,
        'wall_seconds_before_completion':-1.7976931348623157e308,
        'peak_sampled_highwater_rss_bytes':99999999999999999999,
        'peak_job_committed_memory_bytes':99999999999999999999,'peak_artifact_name_bytes':99999999999999999999,
        'limits_note':'10ms samples and kernel child-tree committed limit; no unsampled RSS/power-loss guarantee',
        'record_sha256':'f'*64}
    if len(canonical(completion_shape))+1>COMPLETION_BYTES:
        raise Refusal('completion_bound_before_compute')
    ceiling=record_size_ceiling()
    evidence_cap=LIMITS['evidence_bytes'] if evidence_cap is None else evidence_cap
    artifact_cap=LIMITS['artifact_bytes'] if artifact_cap is None else artifact_cap
    if ceiling['evidence_upper_bound']>evidence_cap:
        raise Refusal('full_evidence_bound_before_compute')
    if ceiling['aggregate_upper_bound']>artifact_cap:
        raise Refusal('aggregate_artifact_bound_before_compute')
    return {'schema':'dense-upright-size-envelope-v2','frame_site_limits':[256,4096],
        'metadata_bytes':metadata,'metadata_cap':STATIC_METADATA_BYTES,
        'evidence_upper_bound':ceiling['evidence_upper_bound'],'completion_upper_bound':COMPLETION_BYTES,
        'log_cap':LOG_BYTES,'aggregate_upper_bound':ceiling['aggregate_upper_bound'],
        'evidence_cap':LIMITS['evidence_bytes'],'artifact_cap':LIMITS['artifact_bytes'],
        'method':'canonical maximum-shaped record + three bounded static-metadata expansions'}


def budget_check(elapsed, rss, artifact_bytes, directory=None):
    if elapsed>LIMITS['wall_seconds']:
        raise Refusal('wall_time_limit')
    if rss>LIMITS['job_committed_bytes']:
        raise Refusal('rss_limit')
    if artifact_bytes>LIMITS['artifact_bytes']:
        raise Refusal('artifact_byte_limit')
    if directory is not None:
        log=Path(directory)/'worker.log'
        if log.exists() and log.stat().st_size>LOG_BYTES:
            raise Refusal('reserved_log_byte_limit')


def publish(path, record, validator, *, remove_failed_completion=False):
    """Local20MiB envelope; same staged/fsync/no-clobber hardlink behavior."""
    path=Path(path)
    if path.exists():
        raise FileExistsError(str(path))
    encoded=canonical(record)+b'\n'
    is_completion=record.get('schema')=='dense-upright-sift-ambiguity-completion-v2'
    cap=COMPLETION_BYTES if is_completion else LIMITS['evidence_bytes']
    if len(encoded)>cap:
        raise Refusal('publication_record_byte_limit')
    infrastructure=_infrastructure()
    current=infrastructure.artifact_size(path.parent)
    budget_check(0,0,current,path.parent)
    if current+2*len(encoded)>LIMITS['artifact_bytes']:
        raise Refusal('aggregate_publication_byte_limit')
    if not is_completion:
        validate_bound=record['size_bound']
        if (len(encoded)>validate_bound['evidence_upper_bound']
                or max(2*len(encoded)+LOG_BYTES,len(encoded)+2*COMPLETION_BYTES+LOG_BYTES)>validate_bound['aggregate_upper_bound']):
            raise Refusal('publication_size_envelope')
    temporary=path.with_name('.'+path.name+'.partial');created=linked=False
    def readback(target):
        data=read(target,cap)
        if data!=encoded:
            raise Refusal('publication_readback_bytes')
        validator(sealed(data))
    try:
        with temporary.open('xb') as stream:
            created=True
            if stream.write(encoded)!=len(encoded):
                raise Refusal('short_write')
            stream.flush();os.fsync(stream.fileno())
        readback(temporary)
        os.link(temporary,path);linked=True
        budget_check(0,0,infrastructure.artifact_size(path.parent),path.parent)
        readback(path)
    except BaseException:
        if remove_failed_completion and linked:path.unlink()
        raise
    finally:
        if created:temporary.unlink()
    return {'path':path.name,'sha256':sha(encoded),'bytes':len(encoded)}


def finalize(body):
    record = {**body, "record_sha256": sha(canonical(body))}
    if len(canonical(record)) + 1 > LIMITS["evidence_bytes"]:
        raise Refusal("evidence_byte_limit")
    return record


def sealed(data):
    record = parse(data)
    body = dict(record)
    if canonical(record) + b"\n" != data or body.pop("record_sha256", None) != sha(canonical(body)):
        raise Refusal("source_seal")
    return record


def _infrastructure():
    sys.path.insert(0, str(ROOT))
    from scripts.evaluation import acquire_endpoint_conflict_native_roster
    return acquire_endpoint_conflict_native_roster


def authorize(root=ROOT):
    # First I/O is the independent new authorization, before image imports.
    data = read(root / AUTH, 65536)
    auth = parse(data)
    if (auth.get("disposition") != "ACCEPT_DENSE_UPRIGHT; AUTHORIZE_ONE_AMBIGUITY_RUN"
            or auth.get("single_probe") is not True or auth.get("output") != OUTPUT
            or auth.get("input_pins") != dict(INPUTS)
            or auth.get("diagnostic_sha256") != DIAGNOSTIC_SHA
            or auth.get("treatment") != TREATMENT or auth.get("limits") != LIMITS):
        raise Refusal("independent_dense_authorization_required")
    pins = {**SOURCE_PINS, PROBE: auth.get("code_pins", {}).get(PROBE),
            TEST: auth.get("code_pins", {}).get(TEST)}
    for name, pin in pins.items():
        if type(pin) is not str or len(pin) != 64:
            raise Refusal("authorization_code_pins_required")
        pinned(root / name, pin)
    record = sealed(pinned(root / DIAGNOSTIC, DIAGNOSTIC_SHA, 1048576))
    entries = record["source_region_masks"]["entries"]
    if len(entries) != 3:
        raise Refusal("exact_three_masks")
    for entry, (endpoint, region, packed, raw) in zip(entries, MASK_PINS):
        if (tuple(entry["endpoint"]) != endpoint or entry["mask"]["source_region_id"] != region
                or entry["mask"]["mask_sha256"] != packed
                or entry["full_native_bool_c_sha256"] != raw):
            raise Refusal("exact_source_mask_binding")
    ingress = record["ingress"]
    if len(ingress) != 2:
        raise Refusal("exact_two_native_frames")
    for pin, (name, input_sha), frame in zip(ingress, INPUTS, ("image-0000", "image-0001")):
        if (pin["frame_id"] != frame or pin["byte_sha256"] != input_sha
                or pin["decoded_shape"] != [900,1600,3] or pin["decoded_dtype"] != "|u1"):
            raise Refusal("fixed_native_ingress")
    return record, {"authorization_sha256": sha(data), "code_pins": pins,
                    "input_pins": dict(INPUTS), "diagnostic_sha256": DIAGNOSTIC_SHA}


def _np():
    import numpy
    return numpy


def _roster():
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))
    from scripts.evaluation import image_point_candidate_roster
    return image_point_candidate_roster


@dataclass
class Budget:
    limit: int
    requested: int = 0
    charged: int = 0
    processed: int = 0

    def reserve(self, units):
        if type(units) is not int or units < 0:
            raise Refusal("invalid_work_units")
        self.requested += units
        if self.charged + units > self.limit:
            raise Refusal("work_budget_overflow")
        self.charged += units

    def close(self):
        return {"requested":self.requested,"charged":self.charged,"processed":self.processed,
                "uncharged":self.requested-self.charged,"unprocessed":self.charged-self.processed}


def plan_sites(frames, masks):
    """All region/usable membership and BOTH site caps precede descriptor compute."""
    roster = _roster()
    if (len(frames) != 2 or [f.frame_id for f in frames] != ["image-0000","image-0001"]
            or len(masks) != 3 or [m.frame_id for m in masks] != ["image-0000","image-0000","image-0001"]):
        raise Refusal("exact_frame_mask_layout")
    if asdict(roster.SETTINGS)["mask_boundary_pad_px"] != 0.0:
        raise Refusal("source_footprint_policy_changed")
    sources = [roster._frame_source(frame) for frame in frames]
    mask_records = [roster._region_record(mask, frames[0 if i < 2 else 1]) for i,mask in enumerate(masks)]
    shapes = [frame.pixels.shape[:2] for frame in frames]
    if sum(h*w for h,w in shapes) > LIMITS["decoded_pixels"]:
        raise Refusal("decoded_pixel_limit")
    sizes = [len(range(8,w,16))*len(range(8,h,16)) for h,w in shapes]
    centers = sum(sizes)
    if centers > LIMITS["grid_centers"]:
        raise Refusal("grid_center_limit")
    work = Budget(LIMITS["mask_memberships"])
    work.reserve(2*sizes[0]+sizes[1])
    eligible, excluded = [[],[],[]], [0,0,0]
    for frame_index, frame in enumerate(frames):
        h,w = shapes[frame_index]
        region_indices = (0,1) if frame_index == 0 else (2,)
        for y in range(8,h,16):
            for x in range(8,w,16):
                for region_index in region_indices:
                    work.processed += 1
                    if masks[region_index].pixels[y,x]:
                        # Accepted production pad==0 short-circuits footprint keep.
                        # Center usable mask remains authoritative; no erosion.
                        if frame.usable_mask[y,x]:
                            eligible[region_index].append([x,y])
                        else:
                            excluded[region_index] += 1
    sites = [sorted({tuple(p) for points in eligible[:2] for p in points}, key=lambda p:(p[1],p[0])),
             [tuple(p) for p in eligible[2]]]
    if (any(len(points)>frame_site_cap(i) for i,points in enumerate(sites))
            or sum(len(points) for points in sites)>LIMITS['total_descriptor_sites']):
        raise Refusal("descriptor_site_limit_before_compute")
    return {"sources":sources,"mask_sources":mask_records,"shapes":[list(s) for s in shapes],
            "grid_centers":centers,"grid_centers_per_frame":sizes,"mask_work":work.close(),
            "eligible_sites":eligible,"usable_excluded_sites":excluded,
            "sites":[[list(p) for p in points] for points in sites],
            "shared_left_site_duplicates":len(eligible[0])+len(eligible[1])-len(sites[0]),
            "footprint_policy":"production pad0: no footprint erosion; usable center only"}


def general_position(points):
    points = sorted(set(tuple(p) for p in points))
    if len(points)<4:
        return False
    points = [(Fraction(x),Fraction(y)) for x,y in points]
    def area(a,b,c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    a,b=points[:2]
    c=next((p for p in points[2:] if area(a,b,p)),None)
    if c is None:
        return False
    triangle=(a,b,c);edge_points={}
    for p in points:
        if p in triangle:
            continue
        edges=[i for i,(u,v) in enumerate(((a,b),(b,c),(c,a))) if not area(u,v,p)]
        if not edges:
            return True
        edge_points.setdefault(edges[0],p)
        if len(edge_points)>=2:
            witnesses=triangle+tuple(edge_points.values())
            return any(all(area(*t) for t in itertools.combinations(q,3))
                       for q in itertools.combinations(witnesses,4))
    return False


def good_four(points):
    if len(points)!=4 or len({tuple(p) for p in points})!=4:
        return False
    normalized=[]
    for x,y in points:
        if not math.isfinite(x) or not math.isfinite(y):
            raise Refusal("nonfinite_witness_coordinate")
        normalized.append((Fraction.from_float(float(x)),Fraction.from_float(float(y))))
    determinants=[(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
                  for a,b,c in itertools.combinations(normalized,3)]
    return all(value!=0 for value in determinants)


def common_four(pairs, budget, determinant_budget=None, source_binding_sha256=None):
    """Deterministic common-pair witnesses; capped work is explicitly unknown."""
    if len(pairs)>1024 or len({p['pair_id'] for p in pairs})!=len(pairs):
        raise Refusal("common_witness_match_limit_or_identity")
    determinant_budget=determinant_budget or Budget(budget.limit*8)
    before=budget.charged;before_determinants=determinant_budget.charged
    ordered=sorted(pairs,key=lambda p:p["pair_id"])
    def result(status,quad=()):
        return {"status":status,"work_nodes":budget.charged-before,
                "determinants":determinant_budget.charged-before_determinants,
                "pair_ids":[p['pair_id'] for p in quad],
                "coordinates":[{"pair_id":p['pair_id'],"left_id":p['left_id'],"right_id":p['right_id'],
                                "left_xy":p['left_xy'],"right_xy":p['right_xy']} for p in quad],
                "source_binding_sha256":source_binding_sha256}
    for quad in itertools.combinations(ordered,4):
        if budget.charged == budget.limit:
            return result("budget_inconclusive")
        budget.reserve(1)
        valid=False
        if (len({p['left_id'] for p in quad})==4 and len({p['right_id'] for p in quad})==4
                and len({tuple(p['left_xy']) for p in quad})==4
                and len({tuple(p['right_xy']) for p in quad})==4):
            determinant_budget.reserve(8)
            left_valid=good_four([p['left_xy'] for p in quad])
            right_valid=good_four([p['right_xy'] for p in quad])
            determinant_budget.processed+=8
            valid=left_valid and right_valid
        budget.processed+=1
        if valid:
            return result("found",quad)
    return result("exhaustive_no_witness")


def inventory(plan, frame_index, returned, runtime):
    np=_np();sites=plan["sites"][frame_index];allowed={tuple(p) for p in sites}
    keys, descriptors=returned
    if len(keys)>len(sites) or len(keys)>frame_site_cap(frame_index):
        raise Refusal("returned_descriptor_site_overflow")
    if descriptors is None:
        if keys:
            raise Refusal("missing_descriptor_table_with_keypoints")
        descriptors=np.empty((0,128),np.float32)
    if (not isinstance(descriptors,np.ndarray) or descriptors.dtype!=np.float32
            or descriptors.shape!=(len(keys),128)):
        raise Refusal("invalid_descriptor_table")
    rows={};invalid_zero=invalid_nonfinite=0
    counts={"planned_sites":len(sites),"returned_rows":len(keys),"missing_sites":0,
            "duplicate_rows":0,"coincident_conflicting_sites":0,"coincident_conflicting_rows":0,
            "zero_rows":0,"nonfinite_rows":0}
    counts['returned_descriptor_table_sha256']=sha(descriptors.tobytes(order='C'))
    counts['returned_keypoints_sha256']=sha(canonical(keys))
    seen=set()
    for key, descriptor in zip(keys,descriptors):
        xy=tuple(key["xy"])
        if (xy not in allowed or key["size"]!=16.0 or key["angle"]!=0.0
                or type(key["site_index"]) is not int or not 0<=key["site_index"]<len(sites)
                or xy!=tuple(sites[key["site_index"]])):
            raise Refusal("returned_prescribed_site_binding")
        seen.add(xy)
        if not np.isfinite(descriptor).all():
            invalid_nonfinite+=1;continue
        if not np.any(descriptor):
            invalid_zero+=1;continue
        rows.setdefault(xy,[]).append(descriptor.copy())
    retained=[];table=[]
    for xy, members in sorted(rows.items(),key=lambda item:(item[0][1],item[0][0])):
        counts["duplicate_rows"]+=len(members)-1
        if any(not np.array_equal(members[0],other) for other in members[1:]):
            counts["coincident_conflicting_sites"]+=1
            counts["coincident_conflicting_rows"]+=len(members);continue
        descriptor=members[0]
        retained.append((xy,len(members)))
        table.append(descriptor)
    array=np.asarray(table,np.float32).reshape(-1,128)
    packed=encode_table(array,frame_index)
    anchor={"schema":INVENTORY,"treatment":TREATMENT,"frame_source":plan["sources"][frame_index],
            "site_plan_sha256":sha(canonical(sites)),"runtime":runtime,
            'descriptor_table_sha256':packed['content_sha256']}
    anchor_sha=sha(canonical(anchor));classes=[]
    for row,(xy,multiplicity) in enumerate(retained):
        content={'xy':list(xy),'size':16.0,'angle':0.0,'descriptor_row':row}
        identifier=sha(canonical({'schema':INVENTORY,'anchor_sha256':anchor_sha,'content':content}))
        classes.append({'feature_id':identifier,'xy':list(xy),'multiplicity':multiplicity,'content':content})
    counts.update(missing_sites=len(sites)-len(seen),zero_rows=invalid_zero,nonfinite_rows=invalid_nonfinite,
                  returned_duplicate_coordinates=len(keys)-len(seen),
                  valid_distinct_sites=len(classes),distinct_coordinates=len({tuple(f['xy']) for f in classes}))
    payload={"anchor":anchor,"anchor_sha256":anchor_sha,"counts":counts,"classes":classes,
             'descriptor_table':packed}
    return payload,array


def distances(left,right,budget):
    np=_np()
    units=len(left)*len(right);budget.reserve(units)  # Before any distance computation.
    result=np.empty((len(left),len(right)),np.float64)
    for start in range(0,len(left),8):
        delta=left[start:start+8,None,:].astype(np.float64)-right[None,:,:].astype(np.float64)
        result[start:start+8]=np.sum(delta*delta,axis=2)
        budget.processed+=len(delta)*len(right)
    if not np.isfinite(result).all():
        raise Refusal("nonfinite_distance")
    return result


def directed(table, origin_indices, target_indices):
    """Squared L2 Lowe ratio, no tie winner and no unavailable second neighbor."""
    accepted={};evidence=[];counts={"origins":len(origin_indices),"absent_second_neighbor":0,
                        "nearest_distance_tie":0,"ratio_rejected":0,"ratio_accepted":0}
    for origin in origin_indices:
        if len(target_indices)<2:
            counts["absent_second_neighbor"]+=1
            evidence.append({"origin":origin,"status":"absent_second_neighbor"});continue
        candidates=sorted((float(table[origin,target]),target) for target in target_indices)
        first,second=candidates[:2]
        if first[0]==second[0]:
            counts["nearest_distance_tie"]+=1;status="nearest_distance_tie"
        elif first[0] < 0.75**2*second[0]:
            accepted[origin]=first[1];counts["ratio_accepted"]+=1;status="ratio_accepted"
        else:
            counts["ratio_rejected"]+=1;status="ratio_rejected"
        evidence.append({"origin":origin,"nearest":first[1],"second":second[1],
                         "nearest_squared":first[0],"second_squared":second[0],"status":status})
    return accepted,counts,evidence


def support(features, shape):
    h,w=shape;bins=[[] for _ in range(16)]
    for feature in features:
        x,y=feature["xy"];bins[int(y*4/h)*4+int(x*4/w)].append(feature["feature_id"])
    for group in bins:group.sort()
    by_id={f["feature_id"]:f for f in features};first=[i for group in bins for i in group][:4]
    return {"distinct_sites":len(features),"distinct_coordinates":len({tuple(f['xy']) for f in features}),
            "bins":bins,"occupied_bins":sum(bool(b) for b in bins),
            "all_eligible_general_position":general_position([f["xy"] for f in features]),
            "unchanged_first_four_ids":first,"unchanged_first_four_ready":good_four([by_id[i]["xy"] for i in first])}


def assemble(plan, returned, bindings, runtime, size_bound=None):
    if len(returned)!=2:
        raise Refusal("exact_two_descriptor_tables")
    prepared=prepare_size_bound(plan,bindings,runtime)
    if size_bound is not None and size_bound!=prepared:
        raise Refusal('precompute_size_binding')
    size_bound=prepared
    inventories=[];tables=[]
    for i,table in enumerate(returned):
        payload,array=inventory(plan,i,table,runtime);inventories.append(payload);tables.append(array)
    comparisons=Budget(LIMITS["cross_frame_comparisons"])
    matrix=distances(tables[0],tables[1],comparisons)
    left,right=[inv["classes"] for inv in inventories]
    witnesses=Budget(LIMITS["common_witness_nodes"])
    determinants=Budget(LIMITS['common_witness_determinants']);groups=[]
    for group_index,row in enumerate((176,197)):
        eligible={tuple(p) for p in plan["eligible_sites"][group_index]}
        left_indices=[i for i,f in enumerate(left) if tuple(f["xy"]) in eligible]
        right_indices=list(range(len(right)))
        forward,forward_counts,forward_evidence=directed(matrix,left_indices,right_indices)
        reverse,reverse_counts,reverse_evidence=directed(matrix.T,right_indices,left_indices)
        pairs=[]
        for i,j in sorted(forward.items()):
            if reverse.get(j)==i:
                pair={"left_id":left[i]["feature_id"],"right_id":right[j]["feature_id"],
                      "left_xy":left[i]["xy"],"right_xy":right[j]["xy"],"distance_squared":float(matrix[i,j])}
                pair["pair_id"]=sha(canonical({"schema":SCHEMA,**pair}));pairs.append(pair)
        common=common_four(pairs,witnesses,determinants,sha(canonical(bindings)))
        supply={"left_eligible_sites":len(eligible),"right_eligible_sites":len(plan["eligible_sites"][2]),
                "left":support([left[i] for i in left_indices],plan["shapes"][0]),
                "right":support(right,plan["shapes"][1])}
        if not supply["left"]["all_eligible_general_position"] or not supply["right"]["all_eligible_general_position"]:
            status="insufficient_nondegenerate_supply"
        elif common["status"]=="found":
            status="candidate_discrimination_witness"
        elif common["status"]=="budget_inconclusive":
            status="common_witness_budget_inconclusive"
        else:
            status=("adequate_supply_appearance_ambiguity" if len(pairs)<4
                    else "reciprocal_candidates_without_common_nondegenerate_witness")
        groups.append({"source_row":row,"supply":supply,"discrimination":{"forward":forward_counts,
                       "reverse":reverse_counts,"forward_evidence":forward_evidence,"reverse_evidence":reverse_evidence,
                       "unambiguous_reciprocal_pairs":len(pairs),"pairs":pairs,
                       "distinct_left_endpoints":len({tuple(p['left_xy']) for p in pairs}),
                       "distinct_right_endpoints":len({tuple(p['right_xy']) for p in pairs}),
                       "left_endpoint_general_position":general_position([p["left_xy"] for p in pairs]),
                       "right_endpoint_general_position":general_position([p["right_xy"] for p in pairs]),
                       "common_four_witness":common},"status":status,"physical_association":"unknown"})
    return finalize({"schema":SCHEMA,"status":groups[1]["status"],"treatment":TREATMENT,"limits":LIMITS,
                     "sources":bindings,"plan":plan,"inventories":inventories,"groups":groups,
                     "comparison_work":comparisons.close(),"common_witness_work":witnesses.close(),
                     "common_witness_determinant_work":determinants.close(),
                     'size_bound':size_bound,
                     "physical_association":"unknown","interpretation":"Supply/discrimination only; no accepted association or accuracy."})


def validate(record):
    """Replay table arithmetic/provenance from bounded artifact, never pixels."""
    if len(canonical(record))+1>LIMITS["evidence_bytes"]:
        raise Refusal("evidence_byte_limit")
    body=dict(record)
    if (body.pop("record_sha256",None)!=sha(canonical(body)) or record["schema"]!=SCHEMA
            or record["limits"]!=LIMITS or record["treatment"]!=TREATMENT
            or record["physical_association"]!="unknown"):
        raise Refusal("output_binding")
    plan=record["plan"];np=_np();returned=[]
    if (len(plan['sources'])!=2 or len(plan['mask_sources'])!=3 or len(plan['sites'])!=2
            or len(plan['shapes'])!=2 or len(plan['eligible_sites'])!=3):
        raise Refusal('plan_layout')
    if any(len(shape)!=2 or any(type(v) is not int or v<=0 for v in shape) for shape in plan['shapes']):
        raise Refusal('plan_shape')
    grid_sizes=[len(range(8,w,16))*len(range(8,h,16)) for h,w in plan['shapes']]
    expected_work=2*grid_sizes[0]+grid_sizes[1]
    if (sum(h*w for h,w in plan['shapes'])>LIMITS['decoded_pixels']
            or sum(grid_sizes)>LIMITS['grid_centers'] or plan['grid_centers']!=sum(grid_sizes)
            or plan['grid_centers_per_frame']!=grid_sizes or expected_work>LIMITS['mask_memberships']
            or plan['mask_work']!={'requested':expected_work,'charged':expected_work,'processed':expected_work,'uncharged':0,'unprocessed':0}):
        raise Refusal('plan_budget_binding')
    union=sorted({tuple(p) for group in plan['eligible_sites'][:2] for p in group},key=lambda p:(p[1],p[0]))
    if (plan['sites'][0]!=[list(p) for p in union] or plan['sites'][1]!=plan['eligible_sites'][2]
            or plan['shared_left_site_duplicates']!=sum(len(s) for s in plan['eligible_sites'][:2])-len(union)):
        raise Refusal('plan_site_deduplication')
    for i,group in enumerate(plan['eligible_sites']):
        h,w=plan['shapes'][0 if i<2 else 1]
        if (len({tuple(p) for p in group})!=len(group)
                or any(len(p)!=2 or any(type(v) is not int for v in p) or not (8<=p[0]<w and 8<=p[1]<h)
                       or (p[0]-8)%16 or (p[1]-8)%16 for p in group)):
            raise Refusal('prescribed_grid_binding')
    if len(record['inventories'])!=2 or sum(len(sites) for sites in plan['sites'])>4352:
        raise Refusal('inventory_frame_total_cap')
    for frame_index,inventory_record in enumerate(record["inventories"]):
        keys=[];descriptors=[];sites=plan["sites"][frame_index]
        anchor=inventory_record['anchor'];counts=inventory_record['counts']
        if (len(sites)>frame_site_cap(frame_index) or inventory_record["anchor_sha256"]!=sha(canonical(anchor))
                or anchor['schema']!=INVENTORY or anchor['treatment']!=TREATMENT
                or anchor['frame_source']!=plan['sources'][frame_index]
                or anchor['site_plan_sha256']!=sha(canonical(sites))
                or anchor['descriptor_table_sha256']!=inventory_record['descriptor_table']['content_sha256']):
            raise Refusal("inventory_binding")
        table=decode_table(inventory_record['descriptor_table'],frame_index,inventory_record['classes'])
        for feature in inventory_record["classes"]:
            content=feature["content"]
            if (feature["feature_id"]!=sha(canonical({"schema":INVENTORY,"anchor_sha256":inventory_record["anchor_sha256"],"content":content}))
                    or content["xy"]!=feature["xy"] or content["size"]!=16.0 or content["angle"]!=0.0):
                raise Refusal("feature_content_binding")
            descriptor=table[content['descriptor_row']]
            if descriptor.shape!=(128,) or not np.isfinite(descriptor).all() or not np.any(descriptor):
                raise Refusal("invalid_published_descriptor")
            keys.append({"xy":feature["xy"],"size":16.0,"angle":0.0,"site_index":sites.index(feature["xy"])})
            descriptors.append(descriptor)
        valid=inventory_record['classes'];valid_rows=sum(f['multiplicity'] for f in valid)
        if (len({f['feature_id'] for f in valid})!=len(valid) or len({tuple(f['xy']) for f in valid})!=len(valid)
                or any(type(f['multiplicity']) is not int or f['multiplicity']<1 for f in valid)
                or counts['planned_sites']!=len(sites) or counts['valid_distinct_sites']!=len(valid)
                or counts['distinct_coordinates']!=len(valid) or not 0<=counts['returned_rows']<=len(sites)
                or counts['returned_rows']!=valid_rows+counts['zero_rows']+counts['nonfinite_rows']+counts['coincident_conflicting_rows']
                or counts['returned_duplicate_coordinates']!=counts['returned_rows']-(len(sites)-counts['missing_sites'])):
            raise Refusal('descriptor_count_closure')
        returned.append((keys,np.asarray(descriptors,np.float32).reshape(-1,128)))
    metadata_runtime=record['inventories'][0]['anchor']['runtime']
    if record['inventories'][1]['anchor']['runtime']!=metadata_runtime:
        raise Refusal('runtime_metadata_frame_binding')
    if record['size_bound']!=prepare_size_bound(plan,record['sources'],metadata_runtime):
        raise Refusal('record_size_envelope_binding')
    if len(canonical(record))+1>record['size_bound']['evidence_upper_bound']:
        raise Refusal('proven_evidence_bound')
    # Validate published groups using the already present descriptor classes.
    # Do not recompute distances during readback (no extra charged comparisons).
    for key,limit in (("comparison_work",1048576),("common_witness_work",LIMITS["common_witness_nodes"]),
                      ('common_witness_determinant_work',LIMITS['common_witness_determinants'])):
        work=record[key]
        if not (work["requested"]==work["charged"]==work["processed"]<=limit
                and work["uncharged"]==work["unprocessed"]==0):
            raise Refusal("work_counter_closure")
    if record["comparison_work"]["charged"]!=len(returned[0][0])*len(returned[1][0]):
        raise Refusal("unique_comparison_count")
    classes=[{f["feature_id"]:f for f in inv["classes"]} for inv in record["inventories"]]
    if len(record["groups"])!=2 or record["status"]!=record["groups"][1]["status"]:
        raise Refusal("group_layout")
    for index,group in enumerate(record["groups"]):
        if group["source_row"]!=(176,197)[index] or group["physical_association"]!="unknown":
            raise Refusal("group_provenance")
        ordered_left=record['inventories'][0]['classes'];ordered_right=record['inventories'][1]['classes']
        eligible={tuple(p) for p in plan['eligible_sites'][index]}
        left_indices=[i for i,f in enumerate(ordered_left) if tuple(f['xy']) in eligible]
        right_indices=list(range(len(ordered_right)))
        expected_supply={'left_eligible_sites':len(eligible),'right_eligible_sites':len(plan['eligible_sites'][2]),
            'left':support([ordered_left[i] for i in left_indices],plan['shapes'][0]),
            'right':support(ordered_right,plan['shapes'][1])}
        if group['supply']!=expected_supply:
            raise Refusal('supply_binding')
        discrimination=group['discrimination'];maps=[]
        for direction,origins,targets in (('forward',left_indices,right_indices),('reverse',right_indices,left_indices)):
            evidence=discrimination[direction+'_evidence'];counts=discrimination[direction]
            expected_counts={'origins':len(origins),'absent_second_neighbor':0,'nearest_distance_tie':0,'ratio_rejected':0,'ratio_accepted':0}
            if [entry['origin'] for entry in evidence]!=origins:
                raise Refusal('directed_evidence_domain')
            admitted={}
            for entry in evidence:
                if len(targets)<2:
                    expected_status='absent_second_neighbor'
                else:
                    d0,d1=entry['nearest_squared'],entry['second_squared']
                    if (entry['nearest'] not in targets or entry['second'] not in targets or entry['nearest']==entry['second']
                            or not math.isfinite(d0) or not math.isfinite(d1) or not 0<=d0<=d1):
                        raise Refusal('directed_nearest_binding')
                    expected_status='nearest_distance_tie' if d0==d1 else ('ratio_accepted' if d0<.75**2*d1 else 'ratio_rejected')
                    if expected_status=='ratio_accepted':admitted[entry['origin']]=entry['nearest']
                expected_counts[expected_status]+=1
                if entry['status']!=expected_status:
                    raise Refusal('directed_ratio_binding')
            if counts!=expected_counts:
                raise Refusal('directed_count_closure')
            maps.append(admitted)
        pairs=discrimination["pairs"]
        expected_edges={(ordered_left[i]['feature_id'],ordered_right[j]['feature_id'])
                        for i,j in maps[0].items() if maps[1].get(j)==i}
        if {(p['left_id'],p['right_id']) for p in pairs}!=expected_edges or len(pairs)!=len(expected_edges):
            raise Refusal('reciprocal_pair_binding')
        if len(pairs)!=group["discrimination"]["unambiguous_reciprocal_pairs"]:
            raise Refusal("pair_count")
        for pair in pairs:
            pbody=dict(pair)
            if pbody.pop("pair_id")!=sha(canonical({"schema":SCHEMA,**pbody})):
                raise Refusal("pair_binding")
            if (classes[0][pair["left_id"]]["xy"]!=pair["left_xy"]
                    or classes[1][pair["right_id"]]["xy"]!=pair["right_xy"]):
                raise Refusal("pair_endpoint_binding")
        witness=group["discrimination"]["common_four_witness"]
        if (discrimination['distinct_left_endpoints']!=len({tuple(p['left_xy']) for p in pairs})
                or discrimination['distinct_right_endpoints']!=len({tuple(p['right_xy']) for p in pairs})
                or discrimination['left_endpoint_general_position']!=general_position([p['left_xy'] for p in pairs])
                or discrimination['right_endpoint_general_position']!=general_position([p['right_xy'] for p in pairs])):
            raise Refusal('candidate_endpoint_binding')
        if witness['source_binding_sha256']!=sha(canonical(record['sources'])):
            raise Refusal('witness_source_binding')
        if witness["status"]=="found":
            by_id={p["pair_id"]:p for p in pairs};quad=[by_id[i] for i in witness["pair_ids"]]
            if not (len({p['left_id'] for p in quad})==len({p['right_id'] for p in quad})==4
                    and good_four([p['left_xy'] for p in quad]) and good_four([p['right_xy'] for p in quad])):
                raise Refusal("common_witness_binding")
            expected_coordinates=[{key:p[key] for key in ('pair_id','left_id','right_id','left_xy','right_xy')} for p in quad]
            if witness['coordinates']!=expected_coordinates:
                raise Refusal('witness_coordinate_binding')
        elif (witness["status"] not in ("exhaustive_no_witness","budget_inconclusive")
              or witness["pair_ids"] or witness['coordinates']):
            raise Refusal("unknown_witness_status")
        if witness['status']=='exhaustive_no_witness' and witness['work_nodes']!=math.comb(len(pairs),4):
            raise Refusal('nonexhaustive_no_witness')
        if (witness['status']=='budget_inconclusive'
                and (record['common_witness_work']['charged']!=LIMITS['common_witness_nodes']
                     or witness['work_nodes']>=math.comb(len(pairs),4))):
            raise Refusal('false_budget_inconclusive')
        expected_status=('insufficient_nondegenerate_supply' if not expected_supply['left']['all_eligible_general_position']
            or not expected_supply['right']['all_eligible_general_position'] else
            ('candidate_discrimination_witness' if witness['status']=='found' else
             ('common_witness_budget_inconclusive' if witness['status']=='budget_inconclusive' else
              ('adequate_supply_appearance_ambiguity' if len(pairs)<4 else 'reciprocal_candidates_without_common_nondegenerate_witness'))))
        if group['status']!=expected_status:
            raise Refusal('group_status_binding')
    if (sum(g['discrimination']['common_four_witness']['work_nodes'] for g in record['groups'])!=record['common_witness_work']['charged']
            or sum(g['discrimination']['common_four_witness']['determinants'] for g in record['groups'])!=record['common_witness_determinant_work']['charged']):
        raise Refusal('common_work_group_closure')


def _runtime_descriptors(frame, sites):
    """Actual OpenCV compute only in separately authorized worker; never detect."""
    import cv2
    from mc_photogrammetry.pipeline.correspondence import _to_grey
    if frame.frame_id not in ('image-0000','image-0001'):
        raise Refusal('descriptor_frame_identity')
    if len(sites)>frame_site_cap(0 if frame.frame_id=='image-0000' else 1):
        raise Refusal("descriptor_site_limit_before_compute")
    if not sites:
        return [],_np().empty((0,128),_np().float32)
    gray=_to_grey(frame.pixels)  # Explicit treatment: no zeroing/crop/resize.
    keypoints=[cv2.KeyPoint(float(x),float(y),16.0,0.0,0.0,0,i) for i,(x,y) in enumerate(sites)]
    returned,table=cv2.SIFT_create(nfeatures=512).compute(gray,keypoints)
    keys=[{"xy":list(point.pt),"size":point.size,"angle":point.angle,"site_index":point.class_id} for point in (returned or [])]
    return keys,table


def _worker():
    record,bindings=authorize()
    infrastructure=_infrastructure();services=infrastructure._services()
    services[4]._validate_mask_diagnostic(record)
    buffers=tuple(pinned(ROOT/name,pin,8388608) for name,pin in INPUTS)
    frames=infrastructure.bind_frames(buffers,record["ingress"],services=services)
    groups=infrastructure.bind_groups(record,frames,services=services)
    masks=(groups[0].left,groups[1].left,groups[0].right)
    plan=plan_sites(frames,masks)  # Both frame site caps before ANY compute.
    roster=_roster()
    runtime=roster._runtime(roster.SETTINGS)
    runtime={**runtime,"descriptor_treatment":"SIFT.compute prescribed size16 angle0 octave0; no detector call",
             'preprocess':TREATMENT['preprocessing'],'detector':'no detector; cv2.SIFT_create(nfeatures=512).compute'}
    full_bindings={**bindings,'ingress':record['ingress'],
                   'mask_bundle_sha256':record['source_region_masks']['record_sha256']}
    size_bound=prepare_size_bound(plan,full_bindings,runtime)  # Entire record AND aggregate before compute.
    before=canonical(plan)
    returned=[_runtime_descriptors(frame,sites) for frame,sites in zip(frames,plan["sites"])]
    after_sources=[roster._frame_source(frame) for frame in frames]
    after_masks=[roster._region_record(mask,frames[0 if i<2 else 1]) for i,mask in enumerate(masks)]
    if canonical(after_sources)!=canonical(plan['sources']) or canonical(after_masks)!=canonical(plan['mask_sources']):
        raise Refusal("source_mutated_during_compute")
    # Postflight hashes sources without scanning site membership a second time.
    if canonical(plan)!=before:
        raise Refusal('plan_mutated_during_compute')
    result=assemble(plan,returned,full_bindings,runtime,size_bound)
    after_record,after_bindings=authorize()
    if canonical(after_record)!=canonical(record) or after_bindings!=bindings:
        raise Refusal("postflight_source_changed")
    for name,pin in INPUTS:pinned(ROOT/name,pin,8388608)
    publish(ROOT/OUTPUT/"ambiguity.json",result,validate)


def supervise():
    started=time.monotonic();_,bindings=authorize();directory=ROOT/OUTPUT
    if directory.exists():
        raise Refusal("output_already_exists")
    infrastructure=_infrastructure();job=infrastructure._WindowsJob();process=None
    peak_rss=peak_commit=peak_artifacts=0
    try:
        directory.mkdir(parents=True,exist_ok=False)
        with (directory/"worker.log").open('xb') as log:
            process=subprocess.Popen([sys.executable,'-B','-c',
                'from scripts.evaluation.probe_dense_upright_sift_ambiguity import _worker; _worker()'],
                cwd=ROOT,stdout=log,stderr=log,creationflags=0x4,
                env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1"})
            job.attach_and_resume(process)
            while True:
                rss,commit=job.snapshot();size=infrastructure.artifact_size(directory)
                peak_rss=max(peak_rss,rss);peak_commit=max(peak_commit,commit);peak_artifacts=max(peak_artifacts,size)
                budget_check(time.monotonic()-started,rss,size,directory)
                if process.poll() is not None:break
                time.sleep(.01)
        if process.returncode!=0:raise Refusal("worker_failure: "+str(process.returncode))
        data=read(directory/"ambiguity.json",LIMITS['evidence_bytes']);result=sealed(data);validate(result)
        _,after=authorize()
        if after!=bindings or any(result['sources'][key]!=value for key,value in bindings.items()):
            raise Refusal("completion_source_changed")
        for name,pin in INPUTS:pinned(ROOT/name,pin,8388608)
        completion=finalize({"schema":"dense-upright-sift-ambiguity-completion-v2","status":result['status'],
            "ambiguity_sha256":sha(data),"ambiguity_bytes":len(data),"sources":bindings,"limits":LIMITS,
            "wall_seconds_before_completion":time.monotonic()-started,"peak_sampled_highwater_rss_bytes":peak_rss,
            "peak_job_committed_memory_bytes":peak_commit,"peak_artifact_name_bytes":peak_artifacts,
            "limits_note":"10ms samples and kernel child-tree committed limit; no unsampled RSS/power-loss guarantee"})
        def check(value):
            if canonical(value)!=canonical(completion):raise Refusal("completion_readback")
            budget_check(time.monotonic()-started,job.snapshot()[0],infrastructure.artifact_size(directory),directory)
        publish(directory/"completion.json",completion,check,remove_failed_completion=True)
        print(result['status']+': '+str(directory/'completion.json'))
        return 0 if result['status']=='candidate_discrimination_witness' else 2
    finally:
        job.close()
        if process is not None and process.poll() is None:process.wait(timeout=5)


def main():
    if len(sys.argv)!=1:raise Refusal("fixed_input_no_overrides")
    return supervise()


if __name__=='__main__':
    try:raise SystemExit(main())
    except (Refusal,OSError,ValueError,TypeError,KeyError) as error:
        print('TERMINAL: '+str(error),file=sys.stderr)
        raise SystemExit(1)
