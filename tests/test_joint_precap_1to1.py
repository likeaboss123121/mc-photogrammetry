"""One fixed analytic falsifier. No images/extraction/native CV/renderer/truth."""
import ast
from dataclasses import replace
import math
from pathlib import Path
import sys
import numpy as np
import pytest
from scripts.evaluation import joint_precap_1to1 as experiment
from mc_photogrammetry.pipeline import joint_matching as native


@pytest.fixture(scope='module')
def report():
    return experiment.run_fixed_falsifier()


def test_fixed_packet_outcome(report):
    # Scientific failure is reported NO_GO, never hidden by changing the input.
    assert report['disposition'] in ('ADVANCE_TO_REVIEW','NO_GO')
    expected=report['mechanism'] and report['recovered'] and report['controls_refused']
    assert (report['disposition']=='ADVANCE_TO_REVIEW') is expected
    main=report['comparisons'][0]
    print({'disposition':report['disposition'],'mechanism':report['mechanism'],'recovered':report['recovered'],
        'controls_refused':report['controls_refused'],'edges':len(report['graphs'][0].edges),
        'A':(main[0].result.status,main[0].result.reason,main[0].result.support),
        'B':(main[1].result.status,main[1].result.reason,main[1].result.support),
        'starvation':report['starvation'],'evidence_bytes':report['evidence_bytes']})


def test_fixed_seed_full_proposal_replay_and_caps(report):
    for graph,case in zip(report['graphs'],report['comparisons']):
        assert len(graph.edges)<=512 and graph.truncated==0
        first,second=case
        assert first.graph_sha256==second.graph_sha256 and first.input_sha256==second.input_sha256
        assert experiment._proposal_generation_equal(first,second)
        for arm in case:
            assert arm.result.support_floor==8
            for trace in arm.classes:
                assert trace.generated<=513 and trace.attempts<=2048 and trace.accepted_samples<=512
                assert trace.generated==trace.accepted_samples+1
                assert not trace.proposals.flags.writeable and not trace.rows.flags.writeable
                for order in range(trace.generated):
                    record=trace.proposals[order]
                    residual=native._residuals(trace.rows,trace.norms,record['candidate'])
                    np.testing.assert_array_equal(record['residual'],residual)
                    assert int(record['raw_count'])==np.count_nonzero(residual<=1.)
                    selected=trace.selection_indices(order)
                    assert len(selected)==record['distinct_count']
                    assert len(np.unique(graph.edges[selected,0]))==len(selected)
                    assert len(np.unique(graph.edges[selected,1]))==len(selected)
                    raw=native._greedy_selection(trace.rows,trace.norms,graph,record['candidate'],1.)
                    expected=np.flatnonzero(raw) if raw is not None else np.empty(0,dtype=np.int64)
                    np.testing.assert_array_equal(selected,expected)
    assert report['evidence_bytes']<=experiment.MAX_EVIDENCE_BYTES


def test_A_exact_native_generation_ranking_refit_and_public_decisions(report):
    # Independent native A run verifies code-copy transparency with no patch.
    for packet,graph,case in zip(report['packets'],report['graphs'],report['comparisons']):
        before=dict(native.__dict__)
        actual=native.match_feature_frames(packet.left,packet.right,packet.left_camera,packet.right_camera)
        recorded=case[0].result
        assert actual.status==recorded.status and actual.reason==recorded.reason
        assert actual.hypotheses==recorded.hypotheses
        np.testing.assert_array_equal(actual.matches,recorded.matches)
        assert all(native.__dict__[key] is value for key,value in before.items())
        for trace in case[0].classes:
            original=native._class_modes(trace.rows,trace.norms,graph,experiment.BOUNDS,8)
            retained=[arrays for _,arrays in trace.refinements if arrays is not None]
            assert len(original)==len(retained)
            for output,observed in zip(original,retained):
                for a,b in zip(output,observed):np.testing.assert_array_equal(a,b)


def test_counterexample_starvation_certificate_if_present(report):
    first,second=report['comparisons'][0]
    for item in report['starvation']:
        a=first.classes[item['yaw_class']];b=second.classes[item['yaw_class']]
        for new in item['new_modes']:
            order=new['proposal_order']
            assert b.proposals[order]['decision']==4
            assert all(native._vector_angle_degrees(b.proposals[order]['candidate'],a.proposals[i]['candidate'])>=15.
                for i in a.retained_orders)
    if report['mechanism']:
        assert any(len(item['A_retained'])==3 for item in report['starvation'])


def test_B_expected_analytic_geometry_and_unchanged_gates_if_solved(report):
    value=report['comparisons'][0][1].result
    if not value.solved:
        assert report['disposition']=='NO_GO';return
    assert value.right_yaw_class==0 and value.pose is not None and value.pose.solved
    expected=np.array((.8,-.3,-.15));expected/=np.linalg.norm(expected)
    found=np.asarray(value.pose.baseline_direction)
    angle=math.degrees(math.acos(float(np.clip(found@expected,-1.,1.))))
    assert angle<=1.
    assert value.support>=8 and min(value.spatial_bins_left,value.spatial_bins_right)>=4
    assert value.parallax_degrees>=3. and value.cheirality_fraction>=.75 and value.singular_ratio>=1e-3
    assert value.median_selected_residual_degrees<=1.
    assert value.ambiguity_margin is None or value.ambiguity_margin>=2
    assert value.pose.right_yaw_class==value.right_yaw_class


@pytest.mark.parametrize('index',[1,2,3,4])
def test_predeclared_refusal_controls(report,index):
    for arm in report['comparisons'][index]:
        assert not arm.result.solved,(report['packets'][index].name,arm.arm,arm.result)
        assert arm.result.reason in native.JOINT_REFUSAL_REASONS
        assert arm.result.pose is None or not arm.result.pose.solved


def test_source_pins_no_cv_import_and_no_legacy_harness():
    assert experiment.source_pins()==experiment.EXPECTED_SOURCE
    assert 'cv2' not in sys.modules
    assert 'scripts.evaluation.run_camera_mixed_depth' not in sys.modules
    tree=ast.parse(Path(experiment.__file__).read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            names=[a.name for a in node.names]+([node.module] if isinstance(node,ast.ImportFrom) else [])
            assert not any(any(word in name for word in ('cv2','render','fixture','run_camera')) for name in names if name)
    assert experiment.BOUNDS==native.JointMatchSettings()


def test_exact_public_matcher_code_and_explicit_globals_mapping():
    packet=experiment.fixed_packets()[0];graph=experiment._graph(packet)
    clone=experiment._matcher('B',packet,graph,[])
    assert clone.__code__ is native.match_feature_frames.__code__
    assert clone.__kwdefaults__==native.match_feature_frames.__kwdefaults__
    assert 'detect_features' not in clone.__globals__ and 'match_and_solve_pair' not in clone.__globals__
    assert clone.__globals__['solve_pair'] is native.solve_pair


def test_caps_and_source_drift_refuse_before_geometry(monkeypatch):
    packet=experiment.fixed_packets()[0]
    left=replace(packet.left,points=experiment._readonly(np.zeros((129,2))),
        descriptors=experiment._readonly(np.zeros((129,28))))
    with pytest.raises(experiment.FalsifierIntegrityError,match='endpoint_cap'):
        experiment.validate_packet(replace(packet,left=left))
    monkeypatch.setattr(experiment,'EXPECTED_SOURCE',{'not-a-source':'0'*64})
    with pytest.raises(FileNotFoundError):experiment.source_pins()
