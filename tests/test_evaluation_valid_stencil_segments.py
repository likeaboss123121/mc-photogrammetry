"""Independent analytic controls; never invoke a renderer, camera or cv2."""
from dataclasses import replace
import math
import sys
import numpy as np
import pytest

from mc_photogrammetry.reconstruction import evaluation_valid_stencil_segments as valid


def horizontal_step(height=65,width=101,value=255):
    image=np.zeros((height,width),dtype=np.uint8)
    image[height//2:]=value
    return image


def run(gray,mask=None,**caps):
    if mask is None:
        mask=np.zeros(gray.shape,dtype=bool)
    config=replace(valid.FROZEN_CONFIG,**caps)
    return valid.extract_segments(gray,mask,frozen_config=config)


def semantic(result):
    return (result.segments.tobytes(),result.pre_length.tobytes(),result.counts,
            {key:value.tobytes() for key,value in result.arrays.items()})


def test_no_cv2_import_or_metadata_input():
    assert "cv2" not in sys.modules
    assert valid.FROZEN_CONFIG.identifier=="VALID-NATIVE-CHAIN-01"


def test_pointwise_gray_has_independent_integer_color_weights_and_no_input_mutation():
    colors=np.asarray([[[255,0,0],[0,255,0],[0,0,255],[255,255,255],[17,49,101]]],dtype=np.uint8)
    before=colors.copy()
    result=valid.pointwise_gray(colors)
    assert result.tolist()==[[77,149,29,255,(77*17+150*49+29*101+128)//256]]
    assert np.array_equal(colors,before)
    assert not result.flags.writeable


def test_native_horizontal_edge_endpoints_scores_support_and_odd_dimensions():
    gray=horizontal_step()
    before=gray.copy()
    result=run(gray)
    assert result.segments.shape==(1,4)
    assert result.segments[0]==pytest.approx([2.,32.,98.,32.],abs=1e-12)
    assert result.segments.dtype.str=="<f8"
    assert result.counts["edgels"]==97
    assert int(result.arrays["line_ledger"][0]["score"])==97*1020**2
    assert not result.segments.flags.writeable and not result.pre_length.flags.writeable
    assert all(not a.flags.writeable for a in result.arrays.values())
    assert np.array_equal(gray,before)
    assert semantic(result)==semantic(run(gray))


@pytest.mark.parametrize("width,admitted",[(44,0),(45,1),(46,1)])
def test_native_detector_length_boundary_is_inclusive(width,admitted):
    result=run(horizontal_step(width=width))
    assert len(result.pre_length)==1
    assert len(result.segments)==admitted
    assert math.hypot(*(result.pre_length[0,2:]-result.pre_length[0,:2]))==width-5
    assert result.counts["short"]==1-admitted


@pytest.mark.parametrize("contrast,edgels",[(15,0),(16,61),(17,61)])
def test_sobel_absolute_threshold_before_nms(contrast,edgels):
    gray=horizontal_step(height=45,width=65,value=contrast)
    result=run(gray)
    # Independent step response: Sobel=4*contrast, squared threshold=64^2.
    assert result.counts["edgels"]==edgels


def test_excluded_adversarial_rgb_values_cannot_create_constant_valid_edges():
    h,w=47,91
    mask=np.zeros((h,w),dtype=bool)
    mask[13:31,22:69]=True
    first=np.full((h,w),83,dtype=np.uint8)
    other=first.copy()
    rng=np.random.default_rng(410)
    other[mask]=rng.integers(0,256,size=mask.sum(),dtype=np.uint8)
    a,b=run(first,mask),run(other,mask)
    assert semantic(a)==semantic(b)
    assert a.counts["edgels"]==0
    assert len(a.segments)==0


def test_mask_perturbation_preserves_all_candidate_scores_arrays_and_counts():
    gray=horizontal_step(width=181)
    mask=np.zeros(gray.shape,dtype=bool)
    mask[:,87:94]=True
    changed=gray.copy()
    changed[mask]=np.random.default_rng(77).integers(0,256,size=mask.sum(),dtype=np.uint8)
    first,second=run(gray,mask),run(changed,mask)
    assert semantic(first)==semantic(second)
    assert len(first.segments)==2
    assert np.all(first.segments[:,2]-first.segments[:,0]>=40.)
    assert first.segments[0,2]<87 and first.segments[1,0]>93
    # All emitted samples are valid; no permitted missing-edgel or mask jump.
    for row in first.segments:
        samples,outside,_=valid.sample_occupancy(row,mask,mask,
                            valid.AllocationBudget(1_000_000),20_001,endpoint_domain=True)
        assert not outside and np.all(samples==3)


def test_original_mask_and_added_band_footprints_are_separate_with_precedence():
    gray=horizontal_step(width=101)
    original=np.zeros(gray.shape,dtype=bool)
    original[:,45:49]=True
    expanded=original.copy()
    expanded[:,41:53]=True
    result=valid.extract_segments(gray,expanded,original_mask=original)
    assert result.counts["footprint_original"]>0
    assert result.counts["footprint_band"]>0
    assert sum(result.counts[k] for k in ("footprint_frame","footprint_original",
               "footprint_band","footprint_valid"))==gray.size
    assert result.counts["below_threshold"]+result.counts["nms_suppressed"]+result.counts["edgels"]==result.counts["footprint_valid"]
    assert sum(result.counts[k] for k in ("admitted","duplicate","frame","original","band","short"))==len(result.pre_length)


@pytest.mark.parametrize("shape",[(1,1),(3,7),(7,3),(9,11)])
def test_all_invalid_and_small_inputs_are_ordinary_empty_results(shape):
    gray=np.full(shape,200,dtype=np.uint8)
    result=run(gray,np.ones(shape,dtype=bool))
    assert result.segments.shape==(0,4)
    assert result.counts["edgels"]==0


def test_adjacent_parallel_edges_with_disjoint_support_remain_separate():
    gray=np.zeros((65,131),dtype=np.uint8)
    gray[20:24]=255
    result=run(gray)
    assert len(result.segments)==2
    assert result.segments[:,1]==pytest.approx([20.,24.],abs=1e-12)
    sets=[]
    membership=result.arrays["support_membership"]
    for row in result.arrays["line_ledger"]:
        sets.append(set(membership[int(row["offset"]):int(row["offset"])+int(row["count"])]))
    assert not sets[0].intersection(sets[1])
    assert result.counts["duplicate"]==0


@pytest.mark.parametrize("slope,offset",[(1.,20.25),(0.5,20.4),(0.25,20.25)])
def test_analytic_oblique_subpixel_edges_have_real_native_support(slope,offset):
    y,x=np.indices((181,181),dtype=float)
    # A continuous linear transition rasterized independently of the extractor.
    gray=np.clip((y-slope*x-offset)*128+128,0,255).astype(np.uint8)
    result=run(np.ascontiguousarray(gray))
    assert result.counts["edgels"]>0
    assert len(result.segments)>0
    for x0,y0,x1,y1 in result.segments:
        assert abs(y0-slope*x0-offset)<2
        assert abs(y1-slope*x1-offset)<2
        assert math.hypot(x1-x0,y1-y0)>=40


def test_analytic_cross_preserves_both_long_directions_and_edge_accounting():
    y,x=np.indices((161,161))
    gray=np.where((abs(x-80)<=1)|(abs(y-80)<=1),255,0).astype(np.uint8)
    result=run(gray)
    dx=result.segments[:,2]-result.segments[:,0]
    dy=result.segments[:,3]-result.segments[:,1]
    assert np.any(abs(dx)>40) and np.any(abs(dy)>40)
    assert result.counts["graph_edges"]==sum(int(p["count"])-1 for p in result.arrays["paths"])+result.counts["cycle_closures"]
    for i,group in enumerate(result.arrays["fit_groups"]):
        if int(group["parent"])>=0:
            parent=result.arrays["fit_groups"][int(group["parent"])]
            assert int(group["stop"])-int(group["start"])<int(parent["stop"])-int(parent["start"])


@pytest.mark.parametrize("cap",["max_edgels","max_edges","max_neighbor_probes",
    "max_groups","max_path_vertices","max_fit_nodes","max_fit_points",
    "max_candidates","max_projection_samples","max_ledger_bytes"])
def test_typed_caps_fail_closed_without_truncated_extraction(cap):
    with pytest.raises(valid.AcquisitionCapError):
        run(horizontal_step(),**{cap:0})


def test_dense_texture_cap_and_zero_memory_fail_before_buffer_allocation(monkeypatch):
    y,x=np.indices((65,101))
    gray=np.where((x//3+y//11)%2,255,0).astype(np.uint8)
    with pytest.raises(valid.AcquisitionCapError,match="edgel_cap"):
        run(gray,max_edgels=4)
    calls=[]
    original=np.empty
    def empty(*args,**kwargs):
        calls.append(args)
        return original(*args,**kwargs)
    monkeypatch.setattr(valid.np,"empty",empty)
    with pytest.raises(valid.AcquisitionCapError,match="allocation"):
        run(horizontal_step(),max_live_bytes=0)
    assert calls==[]


def test_sobel_neighbors_require_only_declared_valid_derivative_taps():
    # Direct derivative control independent of NMS center eligibility: known ramp.
    gray=np.tile(np.arange(9,dtype=np.uint8),(9,1))
    assert valid._sobel(gray,4,4)==(8,0)
    assert valid._sector(64,0)==(1,0)
    assert valid._sector(0,64)==(0,1)
    assert valid._compatible(np.array((0,0,1020,1020,2*1020**2),dtype=valid.EDGEL),
                             np.array((1,1,1020,-1020,2*1020**2),dtype=valid.EDGEL),1,0) is False


def test_invalid_shapes_masks_superset_and_relaxed_config_are_integrity_errors():
    image=horizontal_step()
    mask=np.zeros(image.shape,dtype=bool)
    with pytest.raises(valid.AcquisitionIntegrityError):
        valid.extract_segments(image[:,::2],mask[:,::2])
    original=mask.copy(); original[20,20]=True
    with pytest.raises(valid.AcquisitionIntegrityError,match="superset"):
        valid.extract_segments(image,mask,original_mask=original)
    with pytest.raises(valid.AcquisitionIntegrityError,match="invalid_cap"):
        run(image,max_edgels=valid.FROZEN_CONFIG.max_edgels+1)


def test_curved_analytic_support_obeys_depth_and_fit_work_caps():
    y,x=np.indices((101,201),dtype=float)
    curve=35+8*np.sin(x/30)
    gray=np.clip((y-curve)*128+128,0,255).astype(np.uint8)
    with pytest.raises(valid.AcquisitionCapError,match="depth_cap"):
        run(np.ascontiguousarray(gray),max_depth=0)
    with pytest.raises(valid.AcquisitionCapError,match="fit_point_cap"):
        run(np.ascontiguousarray(gray),max_fit_points=1)


def test_positive_pure_cycle_has_one_canonical_removed_closure_and_all_edges_once():
    # Independently specified undirected cycle: 0-1-2-3-0.
    neighbors=np.full((4,8),-1,dtype=np.int32)
    neighbors[:,:2]=[[1,3],[0,2],[1,3],[0,2]]
    degree=np.full(4,2,dtype=np.uint8)
    budget=valid.AllocationBudget(valid.FROZEN_CONFIG.max_live_bytes)
    counts={"paths":0,"singletons":0,"cycle_closures":0}
    paths,ids,closures=valid._trace_support_graph(neighbors,degree,budget,valid.FROZEN_CONFIG,counts)
    assert counts["cycle_closures"]==1
    assert closures.view().tolist()==[(0,1)]
    assert ids.view().tolist()==[0,3,2,1]
    assert paths.count==1
    assert sum(int(p["count"])-1 for p in paths.view())+closures.count==4


def test_positive_exact_support_duplicate_preserves_highest_rank_and_distinct_parallel_support():
    coordinates=np.asarray([[1.,2.,41.,2.],[1.,2.,41.,2.],[1.,3.,41.,3.]])
    ledger=np.zeros(3,dtype=valid.LINE)
    ledger["offset"]=[0,3,6]; ledger["count"]=[3,3,3]
    ledger["score"]=[100,200,200]; ledger["node"]=[0,1,2]
    support=np.asarray([2,3,4,2,3,4,7,8,9],dtype=np.uint32)
    budget=valid.AllocationBudget(valid.FROZEN_CONFIG.max_live_bytes)
    order,duplicate,lease=valid._rank_and_deduplicate(coordinates,ledger,support,budget)
    assert order==[1,2,0]
    assert duplicate.tolist()==[True,False,False]
    del order
    budget.release(lease)


@pytest.mark.parametrize("values,sign,reversal",[
    ([0.,0.,1.,1.,2.],1,None),([2.,2.,1.,1.,0.],-1,None),
    ([0.,0.,1.,1.,0.,-1.],1,3),([2.,2.,1.,1.,2.,3.],-1,3),
    ([1.,1.,1.,1.],0,None)])
def test_projection_equal_first_direction_and_first_reversal_are_deterministic(values,sign,reversal):
    state=valid._ProjectionState()
    for i,value in enumerate(values):
        state.add(value,0.5,i)
    assert state.direction==sign and state.reversal==reversal
    assert state.split==0  # Equal maximum residual keeps earliest position.
    if len(values)>2:
        cut=valid._progress_cut(0,len(values),state.reversal,state.split)
        assert 0<cut<len(values)-1
        assert cut+1<len(values) and len(values)-cut<len(values)
