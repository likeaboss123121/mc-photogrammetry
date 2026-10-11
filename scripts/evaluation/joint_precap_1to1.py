"""JOINT-PRECAP-1TO1-01: one fixed pure analytic search-objective falsifier.

Evaluation only. No images, extractor, renderer, truth, native CV or production
patch. Expected geometry is outside proposal ranking. A failed fixed packet is
NO-GO, never a reason to search other seeds/graphs/parameters.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from types import FunctionType
import numpy as np
from mc_photogrammetry.pipeline import joint_matching as native
from mc_photogrammetry.pipeline import connected_camera as ds11
from mc_photogrammetry.pipeline.correspondence import FrameFeatures, ReciprocalGraph
from mc_photogrammetry.pipeline.contracts import CameraEstimate, CameraIntrinsics, CameraPose

ROOT = Path(__file__).resolve().parents[2]
DESIGN = 'JOINT-PRECAP-1TO1-01'
MAX_ENDPOINTS = 128
MAX_EDGES = 512
MAX_PROPOSALS = 513
MAX_EVIDENCE_BYTES = 24*1024*1024
MANAGED_LIMIT = 64*1024*1024
NATIVE_SCRATCH_RESERVE = 32*1024*1024
EXPECTED_SOURCE = {
    'src/mc_photogrammetry/pipeline/joint_matching.py': '9303f56ab194600c58a27d97321fa2f2e4323e4bb78276a2df0372e8ed8ae4e3',
    'src/mc_photogrammetry/pipeline/connected_camera.py': '3b5df735084daddccb7f25511af1b8549c0300d02e4841ade39e39461b8275a6',
    'src/mc_photogrammetry/pipeline/correspondence.py': '192a4851e428d8116e0fbffd28c7ff157d1aedd8035e499c6df7ec2a9d787fa1',
    'src/mc_photogrammetry/pipeline/contracts.py': 'db24e730bfed52ec983ae387a26f1b612fd94abc3e7608053080915c899ed5cc',
    'src/mc_photogrammetry/reconstruction/camera.py': '0515c5bd82f01d3dc174ced6cb54405adef0a755b94c577dae513fc53c4fe680',
}
WIDTH, HEIGHT, FOCAL = 1280, 720, 600.
# These are the source defaults, not user-tunable experiment options.
BOUNDS = native.JointMatchSettings()
GATES = ds11.PairGates()
_DECISIONS = {0:'not_examined_after_cap', 1:'zero_support', 2:'below_floor',
    3:'angular_duplicate', 4:'retained'}


class FalsifierIntegrityError(RuntimeError):
    pass


def _require(value, reason):
    if not value:
        raise FalsifierIntegrityError(reason)


def _sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def source_pins():
    result = {}
    for name in EXPECTED_SOURCE:
        digest=hashlib.sha256()
        with (ROOT/name).open('rb') as stream:
            while chunk := stream.read(65536):digest.update(chunk)
        result[name]=digest.hexdigest()
    _require(result==EXPECTED_SOURCE, 'source_drift')
    return result


def _readonly(value, dtype=np.float64):
    result=np.ascontiguousarray(value,dtype=dtype)
    result.setflags(write=False)
    return result


def _pixels(rays):
    rays=np.asarray(rays,dtype=np.float64)
    return np.column_stack((640.+FOCAL*rays[:,0]/rays[:,2],360.-FOCAL*rays[:,1]/rays[:,2]))


def _camera(yaw=0.):
    return CameraEstimate(CameraIntrinsics((WIDTH,HEIGHT),(FOCAL,FOCAL),(640.,360.)),
        CameraPose(None,yaw,0.,0.))


def _features(name, rays, descriptors):
    return FrameFeatures(name,_readonly(_pixels(rays)),_readonly(descriptors),WIDTH,HEIGHT,WIDTH*HEIGHT)


@dataclass(frozen=True, slots=True)
class AnalyticPacket:
    name: str
    left: FrameFeatures
    right: FrameFeatures
    left_camera: CameraEstimate
    right_camera: CameraEstimate


def fixed_packets():
    """The sole predeclared formula packet and controls; no random case search."""
    minority=np.array([((i%4-1.5)*.38+.027*math.sin(i+1),
        (i//4-1.5)*.23+.019*math.cos(2*i+1),1.) for i in range(16)])
    depths=np.array([4.2+.31*(i%5) for i in range(16)])
    shift=np.array((.8,.3,.15))
    right_minor=minority*depths[:,None]+shift
    eye=np.eye(28)
    left=[];right=[];ld=[];rd=[]
    for group in range(3):
        for row in range(4):
            ray=np.array((-.62+.36*row+.025*group,
                -.36+.23*((2*row+group)%4)+.021*group,1.))
            left.append(ray);ld.append(eye[group*4+row])
            for depth in (4.,6.,8.,10.,12.):
                right.append(depth*ray+np.eye(3)[group]);rd.append(eye[group*4+row])
    left.extend(minority);right.extend(right_minor);ld.extend(eye[12:]);rd.extend(eye[12:])
    camera=_camera()
    main=AnalyticPacket('fixed-starvation',_features('left',left,ld),_features('right',right,rd),camera,camera)
    descriptors=eye[12:]
    identity=AnalyticPacket('identity',_features('left',minority,descriptors),
        _features('right',minority,descriptors),camera,camera)
    # Algebraic ray rotation only: no CameraModel/image/3D renderer fixture.
    theta=math.radians(12.)
    rotation=np.array(((math.cos(theta),0.,-math.sin(theta)),(0.,1.,0.),
        (math.sin(theta),0.,math.cos(theta))))
    rotated=minority@rotation.T
    pure=AnalyticPacket('pure-rotation',_features('left',minority,descriptors),
        _features('right',rotated,descriptors),camera,_camera(12.))
    few=AnalyticPacket('insufficient-support',_features('left',minority[:5],descriptors[:5]),
        _features('right',right_minor[:5],descriptors[:5]),camera,camera)
    other=minority*depths[:,None]+np.array((.8,-.3,.15))
    competition=AnalyticPacket('two-competitive-baselines',
        _features('left',np.vstack((minority,minority)),np.vstack((descriptors,descriptors))),
        _features('right',np.vstack((right_minor,other)),np.vstack((descriptors,descriptors))),camera,camera)
    return (main,identity,pure,few,competition)


def validate_packet(packet):
    _require(isinstance(packet,AnalyticPacket),'packet_type')
    for frame,camera in ((packet.left,packet.left_camera),(packet.right,packet.right_camera)):
        n=len(frame.points)
        _require(0<n<=MAX_ENDPOINTS,'endpoint_cap')
        _require(frame.points.dtype==np.float64 and frame.points.shape==(n,2)
            and frame.descriptors.dtype==np.float64 and frame.descriptors.shape==(n,28)
            and frame.points.flags.c_contiguous and frame.descriptors.flags.c_contiguous
            and not frame.points.flags.writeable and not frame.descriptors.flags.writeable
            and np.isfinite(frame.points).all() and np.isfinite(frame.descriptors).all(),'feature_packet')
        _require(frame.width==WIDTH and frame.height==HEIGHT and frame.boundary_dropped==0
            and frame.sizes is None and frame.usable_pixels==WIDTH*HEIGHT,'frame_metadata')
        _require(camera.intrinsics==_camera().intrinsics and camera.pose is not None
            and camera.pose.position is None and camera.pose.pitch_degrees==0.
            and camera.pose.roll_degrees==0. and camera.pose.yaw_degrees in (0.,12.),'analytic_camera')
    _require(BOUNDS==native.JointMatchSettings() and GATES==ds11.PairGates()
        and BOUNDS.ransac_seed==0 and BOUNDS.max_modes_per_class==3,'default_gate_drift')


def _graph(packet):
    graph=native.reciprocal_topk_graph(packet.left.descriptors,packet.right.descriptors,
        neighbors=BOUNDS.neighbors,max_edges=BOUNDS.max_candidate_edges,
        distance_ceiling=BOUNDS.graph_distance_ceiling)
    _require(len(graph.edges)<=MAX_EDGES and graph.truncated==0 and graph.total_edges==len(graph.edges),'graph_cap')
    return ReciprocalGraph(_readonly(graph.edges,np.int64),_readonly(graph.distances),graph.left_keypoints,
        graph.right_keypoints,graph.total_edges,graph.truncated)


def _ledger_dtype(edges):
    packed=(edges+7)//8
    return np.dtype([('candidate','<f8',(3,)),('sample','<i2',(2,)),('order','<u2'),
        ('raw_count','<u2'),('distinct_count','<u2'),('raw_median','<f8'),('distinct_median','<f8'),
        ('rank','<u2'),('decision','u1'),('refit_index','<i2'),('residual','<f8',(edges,)),
        ('raw_mask','u1',(packed,)),('selection','u1',(packed,))])


@dataclass(frozen=True, slots=True)
class ClassTrace:
    yaw_class: int
    rows: np.ndarray
    norms: np.ndarray
    proposals: np.ndarray
    generated: int
    accepted_samples: int
    attempts: int
    ranked_orders: tuple[int,...]
    retained_orders: tuple[int,...]
    refinements: tuple

    def selection_indices(self, order):
        packed=self.proposals[order]['selection']
        mask=np.unpackbits(packed,bitorder='little')[:len(self.rows)]
        return np.flatnonzero(mask)


@dataclass(frozen=True, slots=True)
class ArmEvidence:
    arm: str
    result: native.JointPairResult
    classes: tuple[ClassTrace,...]
    graph_sha256: str
    input_sha256: str
    evidence_bytes: int


def _class_modes(rows,norms,graph,bounds,support_floor,*,arm,yaw_class,traces):
    """Original generation/refit, with only the declared A/B retention key."""
    edge_count=len(rows)
    _require(arm in ('A','B') and 0<=edge_count<=MAX_EDGES and bounds==BOUNDS,'mode_contract')
    if edge_count<2:
        traces.append(ClassTrace(yaw_class,_readonly(rows.copy()),_readonly(norms.copy()),
            _readonly(np.empty(0,dtype=_ledger_dtype(edge_count)),None),0,0,0,(),(),()))
        return []
    # Exact source generation and attempt/acceptance semantics. Observation of
    # accepted sample IDs makes no RNG call and never enters ranking.
    left_index=graph.edges[:,0];right_index=graph.edges[:,1]
    candidates=[];samples=[]
    all_rows_t=native._null_vector(rows)
    if all_rows_t is not None:candidates.append(all_rows_t);samples.append((-1,-1))
    generator=np.random.default_rng(int(bounds.ransac_seed))
    accepted=0;attempts=0
    max_attempts=max(1,int(bounds.ransac_iterations))*4
    while accepted<int(bounds.ransac_iterations) and attempts<max_attempts:
        attempts+=1
        sample=generator.choice(edge_count,size=2,replace=False)
        first,second=int(sample[0]),int(sample[1])
        if first==second:continue
        if left_index[first]==left_index[second] or right_index[first]==right_index[second]:continue
        sample_rows=rows[sample]
        if not native._sample_rank_ok(sample_rows,bounds.min_singular_ratio):continue
        candidate=native._null_vector(sample_rows)
        if candidate is None:continue
        candidates.append(candidate);samples.append((first,second));accepted+=1
    _require(len(candidates)<=MAX_PROPOSALS and attempts<=2048,'proposal_cap')
    ledger=np.zeros(MAX_PROPOSALS,dtype=_ledger_dtype(edge_count))
    ledger['rank']=65535;ledger['refit_index']=-1
    scored=[]
    for order,candidate in enumerate(candidates):
        residual=native._residuals(rows,norms,candidate)
        mask=residual<=bounds.inlier_residual_degrees
        count=int(np.count_nonzero(mask))
        selection=native._greedy_selection(rows,norms,graph,candidate,bounds.inlier_residual_degrees)
        selected=np.zeros(edge_count,bool) if selection is None else selection
        distinct=int(np.count_nonzero(selected))
        raw_median=float(np.median(residual[mask])) if count else 90.
        distinct_median=float(np.median(residual[selected])) if distinct else 90.
        ledger[order]['candidate']=candidate;ledger[order]['sample']=samples[order]
        ledger[order]['order']=order;ledger[order]['raw_count']=count;ledger[order]['distinct_count']=distinct
        ledger[order]['raw_median']=raw_median;ledger[order]['distinct_median']=distinct_median
        ledger[order]['residual']=residual
        ledger[order]['raw_mask']=np.packbits(mask,bitorder='little')
        ledger[order]['selection']=np.packbits(selected,bitorder='little')
        metric=count if arm=='A' else distinct
        median=raw_median if arm=='A' else distinct_median
        if metric==0:ledger[order]['decision']=1;continue
        scored.append((metric,median,order,candidate))
    scored.sort(key=lambda entry:(-entry[0],entry[1],entry[2]))
    for rank,entry in enumerate(scored):ledger[entry[2]]['rank']=rank
    modes=[];kept=[]
    for count,_median,order,candidate in scored:
        if count<support_floor:ledger[order]['decision']=2;break
        if any(native._vector_angle_degrees(candidate,previous)<bounds.mode_separation_degrees for previous in modes):
            ledger[order]['decision']=3;continue
        ledger[order]['decision']=4;modes.append(candidate);kept.append(order)
        if len(modes)>=int(bounds.max_modes_per_class):break
    results=[];refinements=[]
    for order,candidate in zip(kept,modes):
        refined=native._select_and_refit(rows,norms,graph,candidate,bounds)
        if refined is None:refinements.append((order,None));continue
        t,selection=refined
        residual=native._residuals(rows,norms,t)
        ledger[order]['refit_index']=len(results)
        results.append((t,selection,residual))
        refinements.append((order,(_readonly(t.copy()),_readonly(selection.copy(),np.bool_),_readonly(residual.copy()))))
    ledger.setflags(write=False)
    traces.append(ClassTrace(yaw_class,_readonly(rows.copy()),_readonly(norms.copy()),ledger,len(candidates),
        accepted,attempts,tuple(entry[2] for entry in scored),tuple(kept),tuple(refinements)))
    return results


_MATCH_GLOBALS=('JointMatchSettings','PairGates','reciprocal_topk_graph','select_mutual_matches',
    '_cameras_ok','_refused','_world_to_camera','_camera_rays','_YAW_CLASSES','_ry90','_positive_depth_fraction',
    '_bins_covered','_singular_values','_TINY','_Mode','_median_angle_degrees','_winner_refusal',
    '_score_modes','solve_pair','_solved')


def _matcher(arm,packet,graph,traces):
    """Exact public matcher code, explicit evaluation globals, no production patch."""
    namespace={name:getattr(native,name) for name in _MATCH_GLOBALS}
    namespace.update(np=np,math=math,__name__=__name__)
    def graph_once(left,right,**kwargs):
        _require(left is packet.left.descriptors and right is packet.right.descriptors
            and kwargs==dict(neighbors=BOUNDS.neighbors,max_edges=BOUNDS.max_candidate_edges,
                distance_ceiling=BOUNDS.graph_distance_ceiling),'graph_delegate_binding')
        return graph
    def modes(rows,norms,current,bounds,support_floor):
        _require(current is graph and len(traces)<4,'class_call_order')
        return _class_modes(rows,norms,current,bounds,support_floor,arm=arm,yaw_class=len(traces),traces=traces)
    namespace['reciprocal_topk_graph']=graph_once;namespace['_class_modes']=modes
    function=FunctionType(native.match_feature_frames.__code__,namespace,'evaluation_match_feature_frames',
        native.match_feature_frames.__defaults__)
    function.__kwdefaults__=dict(native.match_feature_frames.__kwdefaults__)
    return function


def _input_sha(packet):
    digest=hashlib.sha256()
    for frame,camera in ((packet.left,packet.left_camera),(packet.right,packet.right_camera)):
        for array in (frame.points,frame.descriptors):digest.update(array.tobytes())
        digest.update(repr(asdict(camera)).encode())
    return digest.hexdigest()


def _graph_sha(graph):
    return _sha_bytes(graph.edges.tobytes()+graph.distances.tobytes())


def compare_packet(packet,*,graph=None):
    source_pins();validate_packet(packet)
    graph=_graph(packet) if graph is None else graph
    _require(isinstance(graph,ReciprocalGraph) and len(graph.edges)<=MAX_EDGES
        and graph.truncated==0 and graph.total_edges==len(graph.edges),'graph_cap')
    # Fixed-size typed ledgers preflight before either arm creates proposals.
    ledger_bound=8*MAX_PROPOSALS*_ledger_dtype(len(graph.edges)).itemsize
    _require(ledger_bound<=MAX_EVIDENCE_BYTES and ledger_bound+NATIVE_SCRATCH_RESERVE<MANAGED_LIMIT,'evidence_cap')
    before=_input_sha(packet);graph_sha=_graph_sha(graph);arms=[]
    for arm in ('A','B'):
        traces=[];matcher=_matcher(arm,packet,graph,traces)
        result=matcher(packet.left,packet.right,packet.left_camera,packet.right_camera,
            matcher=BOUNDS,gates=GATES,ratio_test=.75)
        _require(_input_sha(packet)==before and _graph_sha(graph)==graph_sha,'input_mutation')
        _require(len(traces)==4,'class_trace_complete')
        total=sum(t.rows.nbytes+t.norms.nbytes+t.proposals.nbytes+sum(
            sum(a.nbytes for a in arrays) if arrays is not None else 0 for _,arrays in t.refinements) for t in traces)
        _require(total<=MAX_EVIDENCE_BYTES,'evidence_cap')
        arms.append(ArmEvidence(arm,result,tuple(traces),graph_sha,before,total))
    source_pins()
    return tuple(arms)


def _proposal_generation_equal(first,second):
    return all(a.generated==b.generated and a.accepted_samples==b.accepted_samples and a.attempts==b.attempts
        and all(np.array_equal(a.proposals[:a.generated][field],b.proposals[:b.generated][field])
            for field in ('candidate','sample','order','raw_count','distinct_count','raw_median',
                'distinct_median','residual','raw_mask','selection')) for a,b in zip(first.classes,second.classes))


def diagnose_starvation(first,second):
    """Pure observed retention audit. No expected direction/labels enter ranking."""
    _require(_proposal_generation_equal(first,second),'generation_drift')
    changed=[]
    for a,b in zip(first.classes,second.classes):
        lost=set(a.retained_orders);new=[]
        for order in b.retained_orders:
            t=b.proposals[order]['candidate']
            if not any(native._vector_angle_degrees(t,a.proposals[i]['candidate'])<BOUNDS.mode_separation_degrees
                for i in a.retained_orders):
                new.append({'proposal_order':order,'raw':int(b.proposals[order]['raw_count']),
                    'distinct':int(b.proposals[order]['distinct_count']),
                    'A_rank':int(a.proposals[order]['rank']),'B_rank':int(b.proposals[order]['rank'])})
        if new:changed.append({'yaw_class':a.yaw_class,'A_retained':a.retained_orders,'B_retained':b.retained_orders,
            'new_modes':new,'A_raw':[int(a.proposals[i]['raw_count']) for i in a.retained_orders],
            'A_distinct':[int(a.proposals[i]['distinct_count']) for i in a.retained_orders]})
    return changed


def run_fixed_falsifier():
    packets=fixed_packets();graphs=tuple(_graph(packet) for packet in packets)
    bound=sum(8*MAX_PROPOSALS*_ledger_dtype(len(graph.edges)).itemsize for graph in graphs)
    _require(bound<=MAX_EVIDENCE_BYTES and bound+NATIVE_SCRATCH_RESERVE<MANAGED_LIMIT,'suite_evidence_cap')
    comparisons=tuple(compare_packet(packet,graph=graph) for packet,graph in zip(packets,graphs))
    first,second=comparisons[0]
    starvation=diagnose_starvation(first,second)
    controls_refused=all(not arm.result.solved for case in comparisons[1:] for arm in case)
    recovered=not first.result.solved and second.result.solved
    # At least three distinct raw-heavy retained decoys must evict a stronger
    # endpoint-distinct mode; outcome labels/expected geometry are not read.
    mechanism=any(len(v['A_retained'])==3 and any(new['distinct']>max(v['A_distinct'])
        and new['raw']<min(v['A_raw']) for new in v['new_modes']) for v in starvation)
    disposition='ADVANCE_TO_REVIEW' if mechanism and recovered and controls_refused else 'NO_GO'
    return {'design':DESIGN,'disposition':disposition,'mechanism':mechanism,'recovered':recovered,
        'controls_refused':controls_refused,'starvation':starvation,'packets':packets,'graphs':graphs,
        'comparisons':comparisons,'evidence_bytes':sum(arm.evidence_bytes for case in comparisons for arm in case),
        'source_sha256':source_pins(),'matcher_defaults':asdict(BOUNDS),'pair_gates':asdict(GATES)}
