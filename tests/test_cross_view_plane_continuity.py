"""Source-bound synthetic continuity controls; no seed authority."""
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import weakref

import cv2
import numpy as np
import pytest

from mc_photogrammetry.pipeline.image_derived_connections import (
    build_image_derived_connections, _array_sha256, _MATCH_DIGEST_DOMAIN)
from mc_photogrammetry.reconstruction import region_growth as rg
from mc_photogrammetry.pipeline import image_derived_entry as entry_mod
from mc_photogrammetry.pipeline import image_derived_lattice_measure as measure_mod
from mc_photogrammetry.pipeline import image_only
from mc_photogrammetry.pipeline.contracts import CameraSolution
from mc_photogrammetry.input.media import SourceFrame, MediaKind, MediaManifest
from mc_photogrammetry.reconstruction.calibration import CalibrationResult
from test_image_derived_connections import (
    _simple_session, _solved_result, _budgets, _build_session, _fake_estimate)

_spec = importlib.util.spec_from_file_location("cross_view_plane_consistency", Path(__file__).parents[1]
                                             / "scripts/evaluation/cross_view_plane_consistency.py")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
diagnose = _module.diagnose_plane_union


def _matches(obs, values):
    values = np.asarray(values, np.float64)
    return replace(obs, matches=tuple(tuple(float(v) for v in row) for row in values),
                   source_count=len(values), matches_sha256=_array_sha256(values, _MATCH_DIGEST_DOMAIN))


def _snapshot(snapshot, **changes):
    altered = replace(snapshot, **changes)
    return replace(altered, snapshot_sha256=rg._diagnostic_snapshot_digest(altered))


def _scene(*, folded=False, rotation=False, hole=False, shape=(128, 256), max_side=960):
    """Oblique45deg view of perpendicular Minecraft walls x=4 and z=4.

    Matched planar union is x+z=8. Ray/world truth never enters diagnostics.
    Both fixtures share exact first-frame pixels and source-derived atomization.
    """
    height, width = shape
    rays = np.asarray([(sign*x, y) for sign in (-1, 1)
        for x,y in ((.08,-.16),(.22,-.16),(.22,-.06),(.08,-.06),(.08,-.01),(.22,-.01),(.14,-.12),(.18,-.035))])
    right_axis = np.array((1.,0.,-1.))/np.sqrt(2)
    forward = np.array((1.,0.,1.))/np.sqrt(2)
    directions = rays[:,0,None]*right_axis + rays[:,1,None]*(0.,1.,0.) + forward
    depth = np.full(16, 8/np.sqrt(2))
    if folded:
        depth[:8] = 4/directions[:8,2]; depth[8:] = 4/directions[8:,0]
    world = directions*depth[:,None]
    if folded:
        np.testing.assert_allclose(world[:8,2],4)
        np.testing.assert_allclose(world[8:,0],4)
        assert np.dot((1,0,0),(0,0,1)) == 0
    left = rays*200 + (width/2,height/2)
    moved = world if rotation else world-right_axis
    right = np.column_stack((moved@right_axis/(moved@forward),moved[:,1]/(moved@forward)))*200+(width/2,height/2)
    pixels = np.full(shape,120,np.uint8)
    for x,y in left: cv2.circle(pixels,(int(round(x)),int(round(y))),1,100,-1)
    right_pixels = np.full(shape,120,np.uint8)
    for x,y in right: cv2.circle(right_pixels,(int(round(x)),int(round(y))),1,100,-1)
    mask = np.ones(shape,bool)
    if hole:
        x,y=np.rint(left[1]).astype(int); mask[y-2:y+3,x-2:x+3]=False
    with pytest.MonkeyPatch.context() as patch:
        session=_build_session(patch,["left","right"],
            {key:_fake_estimate((width,height)) for key in ("left","right")},
            images={"left":pixels,"right":right_pixels})
    captured=[]
    def matcher(a,_a,b,_b,**kw):
        result=_solved_result(a,b,support=16)
        result.matches=np.column_stack((left,right)); result.parallax_degrees=0. if rotation else 6.
        return result
    build_image_derived_connections(session,budgets=_budgets(),matcher=matcher,
                                     _correspondence_observer=captured.append)
    config=replace(rg.DEFAULT_REGION_GROWTH_CONFIG,atom_spacing=int(round(height*min(1.,max_side/max(shape)))),
                   slic_iterations=1,slic_compactness=1e6,analysis_max_side=max_side)
    calibration=CalibrationResult(focal_pixels=200.,yaw_degrees=45.,pitch_degrees=0.,
        width=width,height=height,inliers=120,segments=160,median_residual=.5)
    snapshots=[]
    frame=session.frames[0]
    rg.propose_image_planes_v2(frame.image,mask,calibration,config=config,
        _atom_snapshot_observer=snapshots.append,
        _diagnostic_frame_binding=(frame.frame_id,session.session_sha256,frame.image_sha256))
    assert len(snapshots)==1
    return captured[0],snapshots[0]


def test_cross_view_perpendicular_corner_same_first_source():
    plane,snapshot=_scene(); corner,corner_snapshot=_scene(folded=True)
    np.testing.assert_array_equal(np.asarray(plane.matches)[:,:2],np.asarray(corner.matches)[:,:2])
    assert snapshot.pixel_sha256 == corner_snapshot.pixel_sha256
    assert snapshot.labels_bytes == corner_snapshot.labels_bytes
    assert snapshot.usable_bytes == corner_snapshot.usable_bytes
    assert snapshot.working_shape == corner_snapshot.working_shape
    positive,negative=diagnose(plane,snapshot,(0,1)),diagnose(corner,corner_snapshot,(0,1))
    assert positive["status"]=="appearance_only" and positive["source_join"]
    assert positive["heldout_counts"]==(4,4) and positive["max_heldout_error"]<1e-8
    assert negative["status"]=="abstain"
    assert negative["reason"] in ("no_training_projective_model","bilateral_heldout_rejected")
    assert not positive["seed_authority"] and not positive["physical_coplanarity_proven"]
    print("CROSS_VIEW_REPAIR="+json.dumps({"planar":positive,"perpendicular_corner":negative}))


@pytest.mark.parametrize("kind,reason",[("rotation","rotation_or_unmeasured_parallax"),
    ("one_sided","insufficient_bilateral_support"),("insufficient","insufficient_bilateral_support"),
    ("repeated","repeated_texture_or_duplicate_coordinates"),("ambiguity","pair_ambiguity_unresolved"),
    ("refused","pair_not_solved"),("truncated","incomplete_correspondences"),
    ("collinear","no_training_projective_model")])
def test_cross_view_negative_controls(kind,reason):
    obs,snapshot=_scene(rotation=kind=="rotation"); points=np.asarray(obs.matches)
    if kind=="one_sided": points[8:,(0,2)]-=100; obs=_matches(obs,points)
    if kind=="insufficient": obs=_matches(obs,points[:8])
    if kind=="repeated": obs=_matches(obs,np.vstack((np.repeat(points[:1],8,axis=0),np.repeat(points[8:9],8,axis=0))))
    if kind=="ambiguity": obs=replace(obs,ambiguity_margin=0)
    if kind=="refused": obs=replace(obs,status="refused")
    if kind=="truncated": obs=replace(obs,complete=False,source_count=65)
    if kind=="collinear":
        points[:,0]=np.arange(16)*5+90; points[:,2]=points[:,0]-200/(8/np.sqrt(2)); points[:,(1,3)]=60
        obs=_matches(obs,points)
    result=diagnose(obs,snapshot,(0,1))
    assert result["status"]=="abstain" and result["reason"]==reason


def test_cross_view_holdout_only_failure_no_refit():
    obs,snapshot=_scene(); points=np.asarray(obs.matches); points[[9,11,13,15],2]+=8
    result=diagnose(_matches(obs,points),snapshot,(0,1))
    assert result["reason"]=="bilateral_heldout_rejected" and result["fitting_hypotheses"]==36


def test_cross_view_holdout_cannot_resolve_training_models(monkeypatch):
    obs,snapshot=_scene(); points=np.asarray(obs.matches); delta=float(points[0,2]-points[0,0])
    points[1::2,2]+=.6; obs=_matches(obs,points); calls=[]
    def fit(source,target):
        calls.append((source.copy(),target.copy()))
        return np.array(((1.,0.,delta+(.6 if len(calls)==1 else -.6)),(0.,1.,0.),(0.,0.,1.)))
    monkeypatch.setattr(_module,"_fit",fit)
    result=diagnose(obs,snapshot,(0,1))
    assert result["reason"]=="multiple_projective_models" and result["max_heldout_error"] is None
    assert len(calls)==2
    for _,targets in calls:
        assert all(np.any(np.all(points[::2,2:]==target,axis=1)) for target in targets)


def test_cross_view_unique_coordinate_wrong_texture_alias_appearance_only():
    obs,snapshot=_scene(); points=np.asarray(obs.matches); points[:,2]+=7
    obs=_matches(obs,points)
    assert len(np.unique(points[:,2:],axis=0))==16 and obs.ambiguity_margin==3
    result=diagnose(obs,snapshot,(0,1))
    assert result["status"]=="appearance_only" and result["max_heldout_error"]<1e-8
    assert not result["appearance_identity_verified"] and not result["seed_authority"]


@pytest.mark.parametrize("kind,reason",[("frame","source_frame_binding_mismatch"),
    ("session","source_frame_binding_mismatch"),("pixels","source_frame_binding_mismatch"),
    ("rows","pair_coordinate_digest_mismatch"),("mask","atom_snapshot_digest_mismatch"),
    ("labels","atom_snapshot_digest_mismatch"),("scale","source_coordinate_mapping_mismatch"),
    ("adjacency","nonneighboring_atoms")])
def test_cross_view_source_binding_refusals(kind,reason):
    obs,snapshot=_scene()
    if kind=="frame": snapshot=_snapshot(snapshot,frame_id="right")
    if kind=="session": snapshot=_snapshot(snapshot,session_sha256="3"*64)
    if kind=="pixels": snapshot=_snapshot(snapshot,pixel_sha256="3"*64)
    if kind=="rows": obs=replace(obs,matches=obs.matches[::-1])
    if kind=="mask": snapshot=replace(snapshot,usable_bytes=bytes(len(snapshot.usable_bytes)))
    if kind=="labels": snapshot=replace(snapshot,labels_bytes=bytes(len(snapshot.labels_bytes)))
    if kind=="scale": snapshot=_snapshot(snapshot,native_to_working_scale=(2.,1.))
    if kind=="adjacency":
        labels=np.frombuffer(snapshot.labels_bytes,"<i4").reshape(snapshot.working_shape).copy()
        labels[:,129]=2; data=labels.tobytes()
        snapshot=_snapshot(snapshot,labels_bytes=data,labels_sha256=hashlib.sha256(data).hexdigest())
    assert diagnose(obs,snapshot,(0,1))["reason"]==reason


def test_cross_view_actual_mask_hole_and_unequal_rounded_scales():
    obs,snapshot=_scene(hole=True)
    assert diagnose(obs,snapshot,(0,1))["reason"]=="correspondence_in_unusable_atom"
    obs,snapshot=_scene(shape=(131,257),max_side=128)
    assert snapshot.working_shape==(65,128)
    assert snapshot.native_to_working_scale==(257/128,131/65)
    assert snapshot.native_to_working_scale[0]!=snapshot.native_to_working_scale[1]
    assert diagnose(obs,snapshot,(0,1))["status"]=="appearance_only"


@pytest.mark.parametrize("count",[16,64,65])
def test_cross_view_same_pass_capture_parity_cap_cleanup(monkeypatch,count):
    session=_simple_session(monkeypatch,["left","right"]); refs=[];samples=[]
    def matcher(a,_a,b,_b,**kw):
        result=_solved_result(a,b,support=count);refs.append(weakref.ref(result.matches));return result
    baseline=build_image_derived_connections(session,budgets=_budgets(),matcher=matcher)
    actual=build_image_derived_connections(session,budgets=_budgets(),matcher=matcher,_correspondence_observer=samples.append)
    assert actual.to_record()==baseline.to_record()
    assert len(samples[0].matches)==min(count,64) and samples[0].complete is (count<=64)
    assert samples[0].left_pixel_sha256==session.frames[0].image_sha256
    assert all(ref() is None for ref in refs)


def test_cross_view_immutable_snapshot_scalar_result():
    obs,snapshot=_scene(); labels=np.frombuffer(snapshot.labels_bytes,"<i4")
    with pytest.raises(ValueError): labels.setflags(write=True)
    result=diagnose(obs,snapshot,(0,1))
    assert len(json.dumps(result).encode())<1024
    assert not {"matches","points","model","homography"}&result.keys()


def test_cross_view_64_rows_model_cap_never_promotes_found_prefix():
    obs,snapshot=_scene(); points=np.asarray(obs.matches)
    left=np.vstack([points[:8,:2]+(j*.1,j*.2) for j in range(4)]+[points[8:,:2]+(j*.1,j*.2) for j in range(4)])
    obs=_matches(obs,np.column_stack((left,left+(points[0,2]-points[0,0],0))))
    result=diagnose(obs,snapshot,(0,1))
    assert result["hypotheses"]==128 and result["reason"]=="hypothesis_budget_exhausted"
    assert result["max_heldout_error"] is None


def _entry_and_measure(monkeypatch, tmp_path, *, folded=False, hole=False, count=16, observed=True):
    """Actual entry->retained session->lattice hooks, synthetic ingress only.

    Discovery/encoded bytes and persistence are injected; no source image file
    or project payload is read/written. Only camera/feature discovery is stubbed.
    """
    source_obs, _ = _scene(folded=folded, shape=(131,257), max_side=128)
    values=np.asarray(source_obs.matches)
    pixels={}
    for key, cols in (("left",slice(0,2)),("right",slice(2,4))):
        image=np.full((131,257,3),120,np.uint8)
        for x,y in values[:,cols]: cv2.circle(image,(int(round(x)),int(round(y))),1,(100,100,100),-1)
        # Real detected >=40px calibration lines; no invented inlier count.
        for x in range(10,250,18): cv2.line(image,(x,18),(x,115),(60,60,60),1)
        pixels[key]=image
    estimates={key:_fake_estimate((257,131)) for key in pixels}
    estimates={key:replace(value,
        intrinsics=replace(value.intrinsics,focal_length_pixels=(200.,200.)),
        pose=replace(value.pose,yaw_degrees=45.,pitch_degrees=0.)) for key,value in estimates.items()}
    def solve(frames):
        from test_image_derived_connections import _canonical_unresolved
        return CameraSolution(cameras=estimates,unresolved=_canonical_unresolved(list(frames)),scale_known=False)
    monkeypatch.setattr(image_only,"solve_image_cameras",solve)
    def discover(paths,*,recursive,ingress_sink):
        frames=[]
        for key,image in pixels.items():
            frame=SourceFrame(key,str(tmp_path/(key+".png")),MediaKind.IMAGE)
            frames.append(frame)
            ok,data=cv2.imencode(".png",cv2.cvtColor(image,cv2.COLOR_RGB2BGR)); assert ok
            ingress_sink(frame.source_path,frame,data.tobytes())
        return MediaManifest(frames)
    monkeypatch.setattr(entry_mod,"_discover",discover)
    monkeypatch.setattr(entry_mod,"_persist_session",lambda **kw:None)
    samples=[]
    def matcher(a,_a,b,_b,**kw):
        result=_solved_result(a,b,support=count)
        result.matches=values if count==16 else np.resize(values,(count,4))
        return result
    kwargs={} if not observed else {"_correspondence_observer":samples.append}
    entry_function=entry_mod.build_image_derived_entry if not observed else entry_mod._build_image_derived_entry
    entry=entry_function([tmp_path],project_path=tmp_path/"unused.session.json",matcher=matcher,**kwargs)
    real_proposal=rg.propose_image_planes_v2
    config=replace(rg.DEFAULT_REGION_GROWTH_CONFIG,analysis_max_side=128,atom_spacing=65,
                   slic_iterations=1,slic_compactness=1e6)
    def proposal(*args,**kw): return real_proposal(*args,config=config,**kw)
    monkeypatch.setattr(rg,"propose_image_planes_v2",proposal)
    real_gui=measure_mod.create_gui_mask
    if hole:
        def gui(height,width):
            excluded=real_gui(height,width)
            x,y=np.rint(values[1,:2]).astype(int);excluded[y-2:y+3,x-2:x+3]=True
            return excluded
        monkeypatch.setattr(measure_mod,"create_gui_mask",gui)
    snapshots=[]
    kwargs={} if not observed else {"_atom_snapshot_observer":snapshots.append}
    result=measure_mod.measure_image_derived_lattice(session=entry.session,pose_graph=entry.pose_graph,**kwargs)
    return entry,result,samples,snapshots


@pytest.mark.parametrize("folded,hole,count",[(False,False,16),(True,False,16),(False,True,16),(False,False,65)])
def test_cross_view_actual_entry_lattice_same_session_join(monkeypatch,tmp_path,folded,hole,count):
    entry,result,samples,snapshots=_entry_and_measure(monkeypatch,tmp_path,folded=folded,hole=hole,count=count)
    assert len(samples)==1 and len(snapshots)==2
    snapshot=next(s for s in snapshots if s.frame_id==samples[0].left_id)
    assert snapshot.session_sha256==samples[0].session_sha256==entry.session.session_sha256
    assert snapshot.pixel_sha256==entry.session.frames[0].image_sha256
    assert snapshot.native_to_working_scale==(257/128,131/65)
    assert snapshot.source_shape==(131,257,3)
    mask=~measure_mod.create_gui_mask(131,257)
    assert snapshot.native_usable_bytes==mask.tobytes()
    assert snapshot.native_usable_sha256==rg._diagnostic_native_mask_digest("left",(131,257,3),mask.tobytes())
    report=diagnose(samples[0],snapshot,(0,1))
    if count>64: assert report["reason"]=="incomplete_correspondences"
    elif hole: assert report["reason"]=="correspondence_in_unusable_atom"
    elif folded: assert report["status"]=="abstain"
    else: assert report["status"]=="appearance_only" and report["source_join"]
    assert not report["seed_authority"]
    assert not list(tmp_path.iterdir())


def test_cross_view_entry_lattice_complete_hook_default_parity(monkeypatch,tmp_path):
    with pytest.MonkeyPatch.context() as patch:
        baseline,first,_,_=_entry_and_measure(patch,tmp_path,observed=False)
    with pytest.MonkeyPatch.context() as patch:
        actual,second,samples,snapshots=_entry_and_measure(patch,tmp_path,observed=True)
    assert baseline.session.session_sha256==actual.session.session_sha256
    assert baseline.graph.to_record()==actual.graph.to_record()
    assert baseline.pose_graph.to_record()==actual.pose_graph.to_record()
    assert first==second and samples and snapshots


def test_cross_view_missing_snapshot_and_native_gui_digest_tamper():
    obs,snapshot=_scene()
    assert diagnose(obs,None,(0,1))["reason"]=="missing_source_atom_snapshot"
    data=bytearray(snapshot.native_usable_bytes);data[0]^=1
    altered=_snapshot(snapshot,native_usable_bytes=bytes(data))
    assert diagnose(obs,altered,(0,1))["reason"]=="native_gui_mask_digest_mismatch"


def test_cross_view_center_aligned_integer_reduction_changes_boundary_assignment():
    labels=np.zeros((8,16),np.int32);labels[:,9:]=1
    native=np.array(((17.25,9.),(17.5,9.),(19.5,9.)))
    actual=rg._diagnostic_native_to_working(native,(2.,2.))
    np.testing.assert_array_equal(actual,((8.375,4.25),(8.5,4.25),(9.5,4.25)))
    assert rg._diagnostic_atom_membership(labels,actual)==(0,0,1)
    assert rg._diagnostic_atom_membership(labels,native/2.)==(1,1,1)
    # Ties retain the source allocator's existing numpy.rint convention.
    assert tuple(np.rint(actual[:,0]))==(8.,8.,10.)
    small=np.zeros((2,4),np.int32);small[:,2:]=1
    boundary=np.array(((3.2,1.2),))
    corrected=rg._diagnostic_native_to_working(boundary,(2.,2.))
    np.testing.assert_allclose(corrected,((1.35,.35),),rtol=0,atol=1e-15)
    assert rg._diagnostic_atom_membership(small,corrected)==(0,)
    assert rg._diagnostic_atom_membership(small,boundary/2.)==(1,)


def test_cross_view_center_mapping_identity_unequal_rounding_edges_and_holes():
    native=np.array(((0.,0.),(256.,130.),(128.25,61.5)))
    np.testing.assert_array_equal(rg._diagnostic_native_to_working(native,(1.,1.)),native)
    scales=(257/128,131/65)
    mapped=rg._diagnostic_native_to_working(native,scales)
    np.testing.assert_allclose(mapped,(native+.5)/scales-.5,rtol=0,atol=0)
    labels=np.zeros((65,128),np.int32)
    assert rg._diagnostic_atom_membership(labels,mapped[:2])==(-1,-1)
    x,y=np.rint(mapped[2]).astype(int);labels[y,x]=-1
    assert rg._diagnostic_atom_membership(labels,mapped[2:])==(-1,)


def test_cross_view_center_rule_legacy_division_metadata_refused():
    obs,snapshot=_scene()
    old=_snapshot(snapshot,coordinate_rule="native_xy_divided_by_exact_axis_scale_then_numpy_rint")
    assert diagnose(obs,old,(0,1))["reason"]=="source_coordinate_mapping_mismatch"
