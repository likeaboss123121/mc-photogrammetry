"""Frozen synthetic descriptor-table contracts; no image/feature runtime.

Raster arrays and descriptor tables are authored in memory. These controls
verify supply/admission/filtering logic, not OpenCV's descriptor response.
"""
from copy import deepcopy
from dataclasses import asdict, replace
import ast
import base64
import hashlib
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from mc_photogrammetry.pipeline import correspondence
from mc_photogrammetry.input import image
from scripts.evaluation import acquire_endpoint_conflict_native_roster as infrastructure
from scripts.evaluation import image_point_candidate_roster as roster
from scripts.evaluation import probe_dense_upright_sift_ambiguity as subject

QUAD = [[8,8],[24,8],[8,24],[24,24]]
SHIFTED = [[x+16,y] for x,y in QUAD]
RUNTIME = {"kind":"handcrafted_synthetic_descriptor_table","image_compute":False}


@pytest.fixture(autouse=True)
def forbid_image_feature_and_production_matcher_runtime(monkeypatch):
    def forbidden(*args,**kwargs):
        pytest.fail("image/descriptor/detector/matcher/supervisor runtime forbidden")
    monkeypatch.setattr(image,'load_image_bytes',forbidden)
    monkeypatch.setattr(image,'load_image',forbidden)
    monkeypatch.setattr(cv2,'SIFT_create',forbidden)
    for name in ('detect_features','extract_pair_matches','select_mutual_matches','reciprocal_topk_graph','_k_nearest'):
        monkeypatch.setattr(correspondence,name,forbidden)
    for name in ('capture_inventory','acquire_roster','detect_features','_k_nearest'):
        monkeypatch.setattr(roster,name,forbidden)
    for name in ('_runtime_descriptors','_worker','supervise'):
        monkeypatch.setattr(subject,name,forbidden)


def frame(frame_id,height=64,width=64,value=0,unusable=()):
    pixels=np.full((height,width,3),value,np.uint8)
    usable=np.ones((height,width),bool)
    for x,y in unusable:usable[y,x]=False
    pixels.setflags(write=False);usable.setflags(write=False)
    data=b'synthetic-byte-source-not-an-image-'+frame_id.encode()
    transform=roster.identity(width,height,None)
    source={'frame_id':frame_id,'encoded_sha256':roster.digest(data),'encoded_length':len(data),
            'decoded_sha256':roster.array_digest(pixels),'shape':[height,width,3],'dtype':'|u1',
            'mask_sha256':roster.array_digest(usable),'usable_true':True,'transform':transform,
            'domain':roster.DOMAIN,'raster_kind':'native',
            'decoder':"input.image.load_image:exif_transpose+convert('RGB')"}
    return roster.NativeFrame(frame_id,data,pixels,usable,source)


def mask(native,points,label):
    values=np.zeros(native.pixels.shape[:2],bool)
    for x,y in points:values[y,x]=True
    values.setflags(write=False)
    return roster.RegionMask(native.frame_id,native.source['decoded_sha256'],values,
                             roster.array_digest(values),'synthetic exact mask:'+label)


def fixture(left0=QUAD,left1=QUAD,right=SHIFTED,height=64,width=64,unusable=(),value=0):
    frames=(frame('image-0000',height,width,value,unusable),frame('image-0001',height,width,value))
    masks=(mask(frames[0],left0,'row176'),mask(frames[0],left1,'row197'),mask(frames[1],right,'shared-right'))
    return frames,masks,subject.plan_sites(frames,masks)


def returned(plan,index,descriptor_by_xy=None,constant=None,missing=False):
    if missing:return [],None
    sites=plan['sites'][index];keys=[];table=[]
    for slot,xy in enumerate(sites):
        keys.append({'xy':xy,'size':16.0,'angle':0.0,'site_index':slot})
        if descriptor_by_xy is not None:
            descriptor=descriptor_by_xy[tuple(xy)]
        elif constant is not None:
            descriptor=np.full(128,constant,np.float32)
        else:
            descriptor=np.zeros(128,np.float32);descriptor[slot]=100
        table.append(descriptor)
    return keys,np.asarray(table,np.float32).reshape(-1,128)


def packet(left0=QUAD,left1=QUAD,right=SHIFTED,constant=None,value=0):
    _,_,plan=fixture(left0,left1,right,value=value)
    tables=[returned(plan,i,constant=constant) for i in (0,1)]
    return subject.assemble(plan,tables,{'synthetic_only':True},RUNTIME)


def pairs(left,right):
    return [{'pair_id':f'{i:064x}','left_id':f'L{i}','right_id':f'R{i}',
             'left_xy':list(a),'right_xy':list(b),'distance_squared':float(len(left)-i)}
            for i,(a,b) in enumerate(zip(left,right))]


def test_absent_authorization_is_first_io_without_any_services(tmp_path,monkeypatch):
    seen=[];original=subject.read
    def read(path,limit):seen.append(path);return original(path,limit)
    monkeypatch.setattr(subject,'read',read)
    monkeypatch.setattr(subject,'_infrastructure',lambda:pytest.fail('services before auth'))
    with pytest.raises(FileNotFoundError):subject.authorize(tmp_path)
    assert seen==[tmp_path/subject.AUTH]
    assert not (tmp_path/subject.OUTPUT).exists()


def test_invalid_authorization_and_cli_overrides_refuse(tmp_path,monkeypatch):
    path=tmp_path/subject.AUTH;path.parent.mkdir(parents=True)
    path.write_bytes(subject.canonical({'disposition':'not authorized'}))
    with pytest.raises(subject.Refusal,match='authorization_required'):subject.authorize(tmp_path)
    monkeypatch.setattr(subject.sys,'argv',[subject.PROBE,'--image','other.png'])
    with pytest.raises(subject.Refusal,match='fixed_input_no_overrides'):subject.main()


@pytest.mark.parametrize('damage',['none','code','mask','native_shape'])
def test_synthetic_authority_checks_all_fixed_bindings_without_input_read(tmp_path,monkeypatch,damage):
    record={'source_region_masks':{'entries':[{'endpoint':list(endpoint),
        'mask':{'source_region_id':region,'mask_sha256':packed},'full_native_bool_c_sha256':raw}
        for endpoint,region,packed,raw in subject.MASK_PINS]},
        'ingress':[{'frame_id':f'image-{i:04d}','byte_sha256':pin,'decoded_shape':[900,1600,3],
                    'decoded_dtype':'|u1'} for i,(_,pin) in enumerate(subject.INPUTS)]}
    if damage=='mask':record['source_region_masks']['entries'][0]['mask']['source_region_id']='0'*64
    elif damage=='native_shape':record['ingress'][0]['decoded_shape']=[901,1600,3]
    record=subject.finalize(record);data=subject.canonical(record)+b'\n'
    path=tmp_path/subject.DIAGNOSTIC;path.parent.mkdir(parents=True);path.write_bytes(data)
    monkeypatch.setattr(subject,'DIAGNOSTIC_SHA',subject.sha(data));monkeypatch.setattr(subject,'SOURCE_PINS',{})
    pins={}
    for name in (subject.PROBE,subject.TEST):
        code=tmp_path/name;code.parent.mkdir(parents=True,exist_ok=True);code.write_bytes(b'frozen-synthetic-code')
        pins[name]=subject.sha(code.read_bytes())
    auth={'disposition':'ACCEPT_DENSE_UPRIGHT; AUTHORIZE_ONE_AMBIGUITY_RUN','single_probe':True,
          'output':subject.OUTPUT,'input_pins':dict(subject.INPUTS),'diagnostic_sha256':subject.DIAGNOSTIC_SHA,
          'treatment':subject.TREATMENT,'limits':subject.LIMITS,'code_pins':pins}
    apath=tmp_path/subject.AUTH;apath.parent.mkdir(parents=True,exist_ok=True);apath.write_bytes(subject.canonical(auth))
    if damage=='code':(tmp_path/subject.PROBE).write_bytes(b'modified-code')
    if damage=='none':assert subject.authorize(tmp_path)[0]==record
    else:
        with pytest.raises(subject.Refusal):subject.authorize(tmp_path)
    assert not any((tmp_path/name).exists() for name,_ in subject.INPUTS)
    assert not (tmp_path/subject.OUTPUT).exists()


def test_frozen_treatment_source_settings_and_fresh_runtime_paths():
    assert subject.TREATMENT['spacing']==16 and subject.TREATMENT['offset']==8
    assert subject.TREATMENT['size']==16 and subject.TREATMENT['angle']==0
    assert subject.TREATMENT['ratio']==asdict(roster.SETTINGS)['ratio_test']==0.75
    assert subject.TREATMENT['mask_boundary_pad_px']==asdict(roster.SETTINGS)['mask_boundary_pad_px']==0
    assert asdict(roster.SETTINGS)['max_features']==512
    assert not (subject.ROOT/subject.AUTH).exists() and not (subject.ROOT/subject.OUTPUT).exists()
    assert subject.LIMITS['common_witness_nodes']==4096
    assert subject.LIMITS['common_witness_determinants']==32768


@pytest.mark.parametrize('value',[0,42,255])
def test_blank_constant_synthetic_descriptor_evidence_abstains(value):
    result=packet(constant=0,value=value);subject.validate(result)
    for inventory in result['inventories']:
        assert inventory['counts']['zero_rows']==4 and not inventory['classes']
    assert result['comparison_work']['charged']==0
    assert result['status']=='insufficient_nondegenerate_supply'


def test_unique_synthetic_texture_codes_exact_grid_translation_recover_candidates():
    # One-hot descriptor codes model unique tiles; no image descriptor is computed.
    result=packet();subject.validate(result)
    assert result['comparison_work']=={'requested':16,'charged':16,'processed':16,'uncharged':0,'unprocessed':0}
    for group in result['groups']:
        assert group['physical_association']=='unknown'
        assert group['status']=='candidate_discrimination_witness'
        assert group['supply']['left']['unchanged_first_four_ready']
        assert group['discrimination']['unambiguous_reciprocal_pairs']==4
        assert group['discrimination']['common_four_witness']['status']=='found'
        for pair in group['discrimination']['pairs']:
            assert pair['right_xy']==[pair['left_xy'][0]+16,pair['left_xy'][1]]
    assert result['common_witness_work']['charged']==2
    assert result['common_witness_determinant_work']['charged']==16


def test_repeated_identical_synthetic_tiles_ties_abstain_without_order_winner():
    result=packet(constant=1);subject.validate(result)
    for group in result['groups']:
        assert group['supply']['left']['all_eligible_general_position']
        assert group['status']=='adequate_supply_appearance_ambiguity'
        assert group['discrimination']['forward']['nearest_distance_tie']==4
        assert group['discrimination']['reverse']['nearest_distance_tie']==4
        assert not group['discrimination']['pairs']
    assert result['common_witness_work']['charged']==0


def test_distinct_source_masks_identical_appearance_remain_physically_unknown():
    other=[[x+32,y] for x,y in QUAD]
    right=[[x,y+32] for x,y in QUAD]
    _,_,plan=fixture(QUAD,other,right)
    codes=[]
    for i in range(4):
        code=np.zeros(128,np.float32);code[i]=100;codes.append(code)
    lmap={tuple(xy):codes[i] for site in (QUAD,other) for i,xy in enumerate(site)}
    rmap={tuple(xy):codes[i] for i,xy in enumerate(right)}
    result=subject.assemble(plan,[returned(plan,0,lmap),returned(plan,1,rmap)],{'synthetic_only':True},RUNTIME)
    subject.validate(result)
    groups=result['groups']
    assert all(g['status']=='candidate_discrimination_witness' and g['physical_association']=='unknown' for g in groups)
    assert {p['right_id'] for p in groups[0]['discrimination']['pairs']}=={p['right_id'] for p in groups[1]['discrimination']['pairs']}
    assert {p['left_id'] for p in groups[0]['discrimination']['pairs']}.isdisjoint({p['left_id'] for p in groups[1]['discrimination']['pairs']})
    assert result['comparison_work']['charged']==32  # Shared once, not64 across groups.


def test_mask_holes_exact_centers_usable_boundary_and_unchanged_raster():
    centers=QUAD+[[40,8]]
    frames,masks,plan=fixture(centers,QUAD,SHIFTED,unusable=((24,8),),value=42)
    before=frames[0].pixels.tobytes()
    assert not masks[0].pixels[8,7] and masks[0].pixels[8,8]
    assert frames[0].usable_mask[8,8] and not frames[0].usable_mask[8,24]
    assert [8,8] in plan['eligible_sites'][0]  # Neighborhood crosses exact mask boundary.
    assert [24,8] not in plan['eligible_sites'][0]
    assert plan['usable_excluded_sites']==[1,1,0]
    assert [40,8] in plan['eligible_sites'][0] and [40,8] not in plan['eligible_sites'][1]
    assert frames[0].pixels.tobytes()==before and np.all(frames[0].pixels==42)
    keep=correspondence.descriptor_footprint_keep(np.array([[8.,8.]]),np.array([16.]),frames[0].usable_mask,0.)
    assert keep.tolist()==[True]


def test_shared_left_mask_sites_deduplicate_before_descriptor_tables():
    _,_,plan=fixture()
    assert plan['shared_left_site_duplicates']==4
    assert len(plan['sites'][0])==4 and len(plan['sites'][1])==4
    assert plan['grid_centers']==32
    assert plan['mask_work']=={'requested':48,'charged':48,'processed':48,'uncharged':0,'unprocessed':0}


@pytest.mark.parametrize('damage',['frame_order','mask_order','mask_hash','frame_pixels'])
def test_source_layout_and_array_provenance_corruption_refuse(damage):
    frames,masks,_=fixture()
    if damage=='frame_order':frames=frames[::-1]
    elif damage=='mask_order':masks=(masks[2],masks[1],masks[0])
    elif damage=='mask_hash':masks=(replace(masks[0],mask_sha256='0'*64),*masks[1:])
    else:frames=(replace(frames[0],source={**frames[0].source,'decoded_sha256':'0'*64}),frames[1])
    with pytest.raises((subject.Refusal,roster.Refusal)):subject.plan_sites(frames,masks)


def test_missing_descriptors_are_abstention_not_invented_supply():
    _,_,plan=fixture()
    result=subject.assemble(plan,[returned(plan,0,missing=True),returned(plan,1)],{'synthetic_only':True},RUNTIME)
    subject.validate(result)
    assert result['inventories'][0]['counts']['missing_sites']==4
    assert result['comparison_work']['charged']==0
    assert result['status']=='insufficient_nondegenerate_supply'


@pytest.mark.parametrize('kind',['zero','nan','infinity'])
def test_invalid_descriptor_rows_are_explicitly_rejected(kind):
    _,_,plan=fixture();left=returned(plan,0)
    left[1][0]=0 if kind=='zero' else (np.nan if kind=='nan' else np.inf)
    result=subject.assemble(plan,[left,returned(plan,1)],{'synthetic_only':True},RUNTIME)
    subject.validate(result)
    count=result['inventories'][0]['counts']
    assert count['valid_distinct_sites']==3
    assert count['zero_rows']==(1 if kind=='zero' else 0)
    assert count['nonfinite_rows']==(0 if kind=='zero' else 1)
    assert result['status']=='insufficient_nondegenerate_supply'


@pytest.mark.parametrize('conflicting',[False,True])
def test_coincident_coordinates_never_add_independent_support(conflicting):
    _,_,plan=fixture();keys,table=returned(plan,0)
    keys=[keys[0],deepcopy(keys[0]),keys[1]]
    table=np.stack([table[0],table[1] if conflicting else table[0],table[1]])
    result=subject.assemble(plan,[(keys,table),returned(plan,1)],{'synthetic_only':True},RUNTIME)
    subject.validate(result)
    counts=result['inventories'][0]['counts']
    assert counts['returned_duplicate_coordinates']==1
    assert counts['coincident_conflicting_sites']==int(conflicting)
    assert counts['distinct_coordinates']==(1 if conflicting else 2)
    assert result['status']=='insufficient_nondegenerate_supply'


@pytest.mark.parametrize('damage',['xy','size','angle','site_index','dtype','width','missing_table','too_many_rows'])
def test_returned_descriptor_table_prescribed_site_integrity_refuses(damage):
    _,_,plan=fixture();keys,table=returned(plan,0)
    if damage=='xy':keys[0]['xy']=[8.125,8]
    elif damage=='size':keys[0]['size']=17.
    elif damage=='angle':keys[0]['angle']=1.
    elif damage=='site_index':keys[0]['site_index']=1
    elif damage=='dtype':table=table.astype(np.float64)
    elif damage=='width':table=table[:,:127]
    elif damage=='missing_table':table=None
    else:keys.append(deepcopy(keys[0]));table=np.vstack([table,table[0]])
    with pytest.raises(subject.Refusal):subject.assemble(plan,[(keys,table),returned(plan,1)],{'synthetic_only':True},RUNTIME)


def test_second_neighbor_unavailable_in_either_direction_abstains():
    one=[QUAD[0]];result=packet(left0=one,left1=one,right=[SHIFTED[0]])
    subject.validate(result)
    for group in result['groups']:
        assert group['discrimination']['forward']['absent_second_neighbor']==1
        assert group['discrimination']['reverse']['absent_second_neighbor']==1
        assert not group['discrimination']['pairs']


def test_lowe_ratio_strict_squared_threshold_and_tie_rejection():
    table=np.array([[9.,16.],[8.,16.],[0.,0.]],np.float64)
    admitted,counts,_=subject.directed(table,[0,1,2],[0,1])
    assert admitted=={1:0}
    assert counts['ratio_rejected']==counts['nearest_distance_tie']==counts['ratio_accepted']==1
    reversed_admitted,_,_=subject.directed(table,[0,1,2],[1,0])
    assert admitted==reversed_admitted


def test_pixel_and_grid_overflow_refuse_before_any_descriptor_or_distance():
    frames,masks,_=fixture(height=900,width=1600)
    large=(frame('image-0000',901,1600),frame('image-0001',901,1600))
    lm=(mask(large[0],QUAD,'a'),mask(large[0],QUAD,'b'),mask(large[1],SHIFTED,'c'))
    with pytest.raises(subject.Refusal,match='decoded_pixel_limit'):subject.plan_sites(large,lm)
    grid=(frame('image-0000',894,1609),frame('image-0001',894,1609))
    gm=(mask(grid[0],QUAD,'a'),mask(grid[0],QUAD,'b'),mask(grid[1],SHIFTED,'c'))
    with pytest.raises(subject.Refusal,match='grid_center_limit'):subject.plan_sites(grid,gm)
    assert subject.plan_sites(frames,masks)['grid_centers']==11200


def test_asymmetric_site_overflow_refuses_before_compute_and_exact_caps_deduplicate():
    full=[[x,y] for y in range(8,900,16) for x in range(8,1600,16)]
    frames,masks,plan=fixture(full[:256],full[:256],full[:4096],height=900,width=1600)
    assert [len(p) for p in plan['sites']]==[256,4096]
    assert plan['mask_work']['charged']==16800 and plan['shared_left_site_duplicates']==256
    overflow=(mask(frames[0],full[:257],'a'),mask(frames[0],full[:257],'b'),masks[2])
    with pytest.raises(subject.Refusal,match='descriptor_site_limit_before_compute'):subject.plan_sites(frames,overflow)
    right_overflow=(masks[0],masks[1],mask(frames[1],full[:4097],'c'))
    with pytest.raises(subject.Refusal,match='descriptor_site_limit_before_compute'):subject.plan_sites(frames,right_overflow)


def test_membership_and_comparison_budget_refuse_before_bounded_computation(monkeypatch):
    membership=subject.Budget(16800)
    with pytest.raises(subject.Refusal,match='work_budget_overflow'):membership.reserve(16801)
    assert membership.requested==16801 and membership.charged==membership.processed==0
    comparison=subject.Budget(1048576);comparison.reserve(1048576);comparison.processed=1048576
    assert comparison.close()['unprocessed']==0
    with pytest.raises(subject.Refusal):comparison.reserve(1)
    empty=np.zeros((1,128),np.float32)
    monkeypatch.setattr(np,'empty',lambda *args,**kwargs:pytest.fail('allocation before comparison reservation'))
    with pytest.raises(subject.Refusal):subject.distances(empty,empty,subject.Budget(0))


def test_per_view_quartets_do_not_imply_same_four_correspondence_witness():
    left=[(0,1),(1,2),(-1,0),(0,0),(1,0),(2,0)]
    right=[(-1,0),(0,0),(0,1),(1,2),(1,0),(2,0)]
    assert subject.general_position(left) and subject.general_position(right)
    result=subject.common_four(pairs(left,right),subject.Budget(4096))
    assert result['status']=='exhaustive_no_witness' and result['work_nodes']==15
    assert not result['pair_ids'] and not result['coordinates']


def test_common_witness_stable_ids_same_quad_and_exact_eight_determinants():
    candidate=pairs(QUAD,SHIFTED);budget=subject.Budget(4096);det=subject.Budget(32768)
    result=subject.common_four(candidate[::-1],budget,det,'b'*64)
    assert result['status']=='found' and result['pair_ids']==[p['pair_id'] for p in candidate]
    assert result['work_nodes']==1 and result['determinants']==8
    assert budget.close()['processed']==1 and det.close()['processed']==8
    assert result['source_binding_sha256']=='b'*64
    assert [entry['left_xy'] for entry in result['coordinates']]==QUAD


def test_common_search_cap_has_three_explicit_statuses_and_no_materialized_combinations():
    points=[(i,0) for i in range(20)]
    budget=subject.Budget(4096);det=subject.Budget(32768)
    result=subject.common_four(pairs(points,points),budget,det)
    assert result['status']=='budget_inconclusive' and result['work_nodes']==4096
    assert result['determinants']==32768
    assert budget.close()['unprocessed']==det.close()['unprocessed']==0
    five=pairs(points[:5],points[:5])
    assert subject.common_four(five,subject.Budget(5))['status']=='exhaustive_no_witness'
    assert subject.common_four(five,subject.Budget(4))['status']=='budget_inconclusive'
    assert subject.common_four([],subject.Budget(0))['status']=='exhaustive_no_witness'


def test_common_budget_is_global_across_groups_and_capped_before_check():
    budget=subject.Budget(1);det=subject.Budget(8);candidate=pairs(QUAD,SHIFTED)
    assert subject.common_four(candidate,budget,det)['status']=='found'
    assert subject.common_four(candidate,budget,det)['status']=='budget_inconclusive'
    assert budget.charged==budget.processed==1 and det.charged==det.processed==8


@pytest.mark.parametrize('damage',['duplicate_endpoint','coincident_coordinates','nonfinite','oversized'])
def test_common_witness_endpoint_finite_coordinate_and_match_caps(damage):
    candidate=pairs(QUAD,SHIFTED)
    if damage=='duplicate_endpoint':candidate[1]['left_id']=candidate[0]['left_id']
    elif damage=='coincident_coordinates':candidate[1]['right_xy']=candidate[0]['right_xy']
    elif damage=='nonfinite':candidate[1]['left_xy'][0]=float('nan')
    else:candidate=pairs([(i,0) for i in range(1025)],[(i,0) for i in range(1025)])
    if damage in ('nonfinite','oversized'):
        with pytest.raises(subject.Refusal):subject.common_four(candidate,subject.Budget(4096))
    else:assert subject.common_four(candidate,subject.Budget(4096))['status']=='exhaustive_no_witness'


def test_common_search_agrees_with_exhaustive_small_oracle():
    grid=list(itertools.product(range(3),repeat=2))
    for subset in itertools.combinations(grid,5):
        right=[(p[1],p[0]+16) for p in subset]
        expected=any(subject.good_four(q) and subject.good_four([(p[1],p[0]+16) for p in q])
                     for q in itertools.combinations(subset,4))
        observed=subject.common_four(pairs(subset,right),subject.Budget(4096))
        assert (observed['status']=='found')==expected


@pytest.mark.parametrize('damage',['record','class','coordinate','counter','witness','association','treatment','status'])
def test_output_provenance_counter_and_witness_corruption_refuses(damage):
    result=packet()
    if damage=='record':result['record_sha256']='0'*64
    elif damage=='class':result['inventories'][0]['classes'][0]['feature_id']='0'*64
    elif damage=='coordinate':result['inventories'][0]['classes'][0]['xy']=[40,40]
    elif damage=='counter':result['comparison_work']['charged']+=1
    elif damage=='witness':result['groups'][1]['discrimination']['common_four_witness']['coordinates'][0]['left_xy']=[40,40]
    elif damage=='association':result['physical_association']='accepted'
    elif damage=='treatment':result['treatment']={**result['treatment'],'size':17.0}
    else:result['status']='adequate_supply_appearance_ambiguity'
    if damage!='record':result.pop('record_sha256');result=subject.finalize(result)
    with pytest.raises((subject.Refusal,KeyError,ValueError)):subject.validate(result)


def test_evidence_cap_canonical_roundtrip_and_no_clobber(tmp_path):
    overhead=len(subject.canonical(subject.finalize({'pad':''})))+1
    exact=subject.finalize({'pad':'x'*(8388608-overhead)})
    assert len(subject.canonical(exact))+1==8388608
    with pytest.raises(subject.Refusal,match='evidence_byte_limit'):
        subject.finalize({'pad':'x'*(8388608-overhead+1)})
    assert list(tmp_path.iterdir())==[]
    result=packet();path=tmp_path/'ambiguity.json'
    subject.publish(path,result,subject.validate)
    assert subject.canonical(subject.sealed(path.read_bytes()))==subject.canonical(result)
    with pytest.raises(FileExistsError):subject.publish(path,result,subject.validate)
    assert subject.canonical(subject.sealed(path.read_bytes()))==subject.canonical(result)


def test_unchanged_job_limits_and_no_runtime_command_in_this_module():
    assert infrastructure.WALL_SECONDS==subject.LIMITS['wall_seconds']==180
    assert infrastructure.RSS_BYTES==subject.LIMITS['job_committed_bytes']==2147483648
    assert infrastructure.ARTIFACT_BYTES==8388608  # Accepted helper remains unchanged.
    assert subject.LIMITS['artifact_bytes']==20971520
    assert not (subject.ROOT/subject.AUTH).exists() and not (subject.ROOT/subject.OUTPUT).exists()


def test_runtime_source_contract_has_one_compute_unmodified_grayscale_and_no_detector():
    # Source-only inspection; the prohibited runtime function is never invoked.
    module=ast.parse(Path(subject.__file__).read_bytes())
    functions={node.name:node for node in module.body if isinstance(node,ast.FunctionDef)}
    body=functions['_runtime_descriptors']
    calls=[node for node in ast.walk(body) if isinstance(node,ast.Call)]
    attributes=[node.func.attr for node in calls if isinstance(node.func,ast.Attribute)]
    assert attributes.count('compute')==1 and attributes.count('SIFT_create')==1
    assert not any(name in attributes for name in ('detect','detectAndCompute','resize','warpPerspective'))
    gray=[node for node in calls if isinstance(node.func,ast.Name) and node.func.id=='_to_grey']
    assert len(gray)==1 and ast.unparse(gray[0].args[0])=='frame.pixels'
    assert not any(isinstance(node,ast.Assign) and any(isinstance(target,ast.Subscript) for target in node.targets)
                   for node in ast.walk(body))
    worker=functions['_worker'];names=[node.func.id for node in ast.walk(worker)
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)]
    assert names.count('plan_sites')==1  # No second uncharged membership scan.
    prepare=min(node.lineno for node in ast.walk(worker) if isinstance(node,ast.Call)
        and isinstance(node.func,ast.Name) and node.func.id=='prepare_size_bound')
    compute=min(node.lineno for node in ast.walk(worker) if isinstance(node,ast.Call)
        and isinstance(node.func,ast.Name) and node.func.id=='_runtime_descriptors')
    assert prepare<compute  # Full evidence/aggregate envelope precedes both calls.
    assert subject.AUTH.endswith('AUTH-02-2026-10-09.json')
    assert subject.OUTPUT.endswith('-repair-01')


@pytest.mark.parametrize('frame_index,rows',[(0,256),(1,4096)])
def test_lossless_descriptor_table_exact_cap_little_endian_roundtrip(frame_index,rows):
    values=np.arange(rows*128,dtype=np.float32).reshape(rows,128)
    values[0,0]=np.nextafter(np.float32(0),np.float32(1))
    values[0,1]=np.float32(-0.0)  # Preserve every float32 bit, including signed zero.
    packed=subject.encode_table(values,frame_index)
    decoded=subject.decode_table(packed,frame_index)
    raw=values.astype('<f4').tobytes(order='C')
    assert decoded.tobytes(order='C')==raw
    assert packed['row_count']==rows and packed['byte_count']==rows*512
    assert packed['content_sha256']==hashlib.sha256(raw).hexdigest()
    assert packed['data_base64']==base64.b64encode(raw).decode('ascii')
    assert decoded.dtype.str=='<f4' and not decoded.flags.writeable


@pytest.mark.parametrize('frame_index,rows',[(0,257),(1,4097)])
def test_descriptor_table_cap_plus_one_refuses_before_encoding_decoding_allocation(monkeypatch,frame_index,rows):
    values=np.zeros((rows,128),np.float32)
    normal=subject.encode_table(np.ones((1,128),np.float32),frame_index)
    normal.update(row_count=rows,byte_count=rows*512,data_base64='')
    monkeypatch.setattr(subject.base64,'b64encode',lambda *_:pytest.fail('encode before cap refusal'))
    monkeypatch.setattr(subject.base64,'b64decode',lambda *_args,**_kw:pytest.fail('decode before cap refusal'))
    monkeypatch.setattr(np,'frombuffer',lambda *_args,**_kw:pytest.fail('array allocation before cap refusal'))
    with pytest.raises(subject.Refusal,match='row_cap'):subject.encode_table(values,frame_index)
    with pytest.raises(subject.Refusal,match='bound_binding'):subject.decode_table(normal,frame_index)


@pytest.mark.parametrize('damage',['alphabet','padding','pad_bits','truncated','oversized','digest','rows','bytes','dtype','width','order','encoding','nonfinite'])
def test_descriptor_table_strict_canonical_payload_binding_mutations(damage):
    values=np.ones((1,128),np.float32);packed=subject.encode_table(values,0)
    if damage=='alphabet':packed['data_base64']='!'+packed['data_base64'][1:]
    elif damage=='padding':packed['data_base64']=packed['data_base64'][:-1]+'A'
    elif damage=='pad_bits':
        alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
        value=alphabet.index(packed['data_base64'][-2])
        packed['data_base64']=packed['data_base64'][:-2]+alphabet[(value&~3)+1]+'='
    elif damage=='truncated':packed['data_base64']=packed['data_base64'][:-4]
    elif damage=='oversized':packed['data_base64']+='AAAA'
    elif damage=='digest':packed['content_sha256']='0'*64
    elif damage=='rows':packed['row_count']=2
    elif damage=='bytes':packed['byte_count']+=1
    elif damage=='dtype':packed['dtype']='>f4'
    elif damage=='width':packed['width']=127
    elif damage=='order':packed['order']='F'
    elif damage=='encoding':packed['encoding']='base64'
    else:
        values[0,0]=np.nan;raw=values.astype('<f4').tobytes()
        packed['data_base64']=base64.b64encode(raw).decode();packed['content_sha256']=subject.sha(raw)
    with pytest.raises(subject.Refusal):subject.decode_table(packed,0)


@pytest.mark.parametrize('damage',['duplicate','outside','order','boolean','coordinate'])
def test_descriptor_row_references_refuse_before_payload_decode(monkeypatch,damage):
    packed=subject.encode_table(np.ones((2,128),np.float32),0)
    classes=[{'xy':[8,8],'content':{'descriptor_row':0}},{'xy':[24,8],'content':{'descriptor_row':1}}]
    if damage=='duplicate':classes[1]['content']['descriptor_row']=0
    elif damage=='outside':classes[1]['content']['descriptor_row']=2
    elif damage=='order':classes=classes[::-1]
    elif damage=='boolean':classes[0]['content']['descriptor_row']=False
    else:classes[1]['xy']=classes[0]['xy']
    monkeypatch.setattr(subject.base64,'b64decode',lambda *_args,**_kwargs:pytest.fail('payload decode before reference refusal'))
    with pytest.raises(subject.Refusal,match='row_references'):subject.decode_table(packed,0,classes)


def test_descriptor_inventory_canonical_row_order_is_permutation_stable_and_not_per_class_bytes():
    _,_,plan=fixture();keys,values=returned(plan,0)
    first,_=subject.inventory(plan,0,(keys,values),RUNTIME)
    second,_=subject.inventory(plan,0,(keys[::-1],values[::-1]),RUNTIME)
    assert first['descriptor_table']==second['descriptor_table']
    assert first['classes']==second['classes'] and first['anchor']==second['anchor']
    assert [f['content']['descriptor_row'] for f in first['classes']]==[0,1,2,3]
    assert all(set(f['content'])=={'xy','size','angle','descriptor_row'} for f in first['classes'])
    assert first['anchor']['descriptor_table_sha256']==first['descriptor_table']['content_sha256']


@pytest.mark.parametrize('which',['evidence','aggregate'])
def test_full_record_upper_bound_one_byte_over_refuses_before_compute_or_table_work(monkeypatch,which):
    _,_,plan=fixture();ceiling=subject.record_size_ceiling()
    kwargs={'evidence_cap':ceiling['evidence_upper_bound']-1} if which=='evidence' else {'artifact_cap':ceiling['aggregate_upper_bound']-1}
    monkeypatch.setattr(subject,'inventory',lambda *_args,**_kw:pytest.fail('inventory before whole record proof'))
    with pytest.raises(subject.Refusal,match='bound_before_compute'):
        subject.prepare_size_bound(plan,{'synthetic_only':True},RUNTIME,**kwargs)


def test_metadata_and_completion_bounds_refuse_before_descriptor_compute():
    _,_,plan=fixture()
    with pytest.raises(subject.Refusal,match='static_metadata_bound_before_compute'):
        subject.prepare_size_bound(plan,{'pad':'x'*subject.STATIC_METADATA_BYTES},RUNTIME)
    with pytest.raises(subject.Refusal,match='completion_bound_before_compute'):
        subject.prepare_size_bound(plan,{'pad':'x'*subject.COMPLETION_BYTES},RUNTIME)


def test_maximum_count_full_evidence_and_simultaneous_atomic_publication_bound(tmp_path,monkeypatch):
    full=[[x,y] for y in range(8,900,16) for x in range(8,1600,16)]
    _,_,plan=fixture(full[:256],full[:256],full[:4096],height=900,width=1600)
    result=subject.assemble(plan,[returned(plan,0,constant=1),returned(plan,1,constant=1)],{'synthetic_only':True},RUNTIME)
    subject.validate(result)
    assert [inv['counts']['valid_distinct_sites'] for inv in result['inventories']]==[256,4096]
    assert result['comparison_work']=={'requested':1048576,'charged':1048576,'processed':1048576,'uncharged':0,'unprocessed':0}
    assert result['plan']['mask_work']['charged']==16800
    wire=subject.canonical(result)+b'\n';bound=result['size_bound']
    assert len(wire)<=bound['evidence_upper_bound']<=8388608
    assert bound['aggregate_upper_bound']<=20971520
    assert bound['evidence_upper_bound']==7178782 and bound['aggregate_upper_bound']==14423100
    (tmp_path/'worker.log').write_bytes(b'x'*subject.LOG_BYTES)
    actual_link=subject.os.link;peaks=[]
    def link(source,target):
        actual_link(source,target);peaks.append(infrastructure.artifact_size(tmp_path))
    monkeypatch.setattr(subject.os,'link',link)
    subject.publish(tmp_path/'ambiguity.json',result,subject.validate)
    completion=subject.finalize({'schema':'dense-upright-sift-ambiguity-completion-v2','pad':''})
    padding=subject.COMPLETION_BYTES-len(subject.canonical(completion))-1
    completion=subject.finalize({'schema':'dense-upright-sift-ambiguity-completion-v2','pad':'x'*padding})
    subject.publish(tmp_path/'completion.json',completion,lambda value:subject.canonical(value)==subject.canonical(completion))
    assert peaks==[subject.LOG_BYTES+2*len(wire),subject.LOG_BYTES+len(wire)+2*subject.COMPLETION_BYTES]
    assert max(peaks)<=bound['aggregate_upper_bound']
    assert not any(p.name.endswith('.partial') for p in tmp_path.iterdir())
    # Full canonical synthetic observation and receipt are retained in this task's owned run paths.
    receipt_root=subject.ROOT/'agent swarm instantiation template/runs'
    prefix='SOL-DENSE-UPRIGHT-SIFT-BOUNDED-CAP-REPAIR-01'
    with (receipt_root/(prefix+'.maximum-evidence.json')).open('xb') as stream:stream.write(wire)
    sizing={'maximum_record_bytes':len(wire),'maximum_record_sha256':subject.sha(wire),
        'maximum_evidence_upper_bound':bound['evidence_upper_bound'],'maximum_aggregate_upper_bound':bound['aggregate_upper_bound'],
        'actual_atomic_name_peaks':peaks,'log_bytes':subject.LOG_BYTES,'completion_bytes':subject.COMPLETION_BYTES,
        'lead_declared_plan_no_mask_recompute':subject.record_size_ceiling(126,3288,(20,106)),
        'comparison_work':result['comparison_work'],'mask_work':result['plan']['mask_work'],
        'synthetic_only':True,'image_access':0,'sift_runtime':0}
    with (receipt_root/(prefix+'.size-control.json')).open('xb') as stream:
        stream.write((json.dumps(sizing,sort_keys=True,indent=2)+'\n').encode())


def test_log_reservation_and_aggregate_publication_overflow_refuse(tmp_path,monkeypatch):
    result=packet()
    with pytest.raises(subject.Refusal,match='artifact_byte_limit'):
        subject.budget_check(0,0,20971521)
    (tmp_path/'worker.log').write_bytes(b'x'*(subject.LOG_BYTES+1))
    with pytest.raises(subject.Refusal,match='reserved_log_byte_limit'):
        subject.publish(tmp_path/'ambiguity.json',result,subject.validate)
    assert not (tmp_path/'ambiguity.json').exists() and not (tmp_path/'.ambiguity.json.partial').exists()
    (tmp_path/'worker.log').unlink()
    encoded_size=len(subject.canonical(result))+1
    monkeypatch.setattr(infrastructure,'artifact_size',lambda _:20971520-2*encoded_size+1)
    with pytest.raises(subject.Refusal,match='aggregate_publication_byte_limit'):
        subject.publish(tmp_path/'ambiguity.json',result,subject.validate)
