"""Strict diagnostic records/publication and generated-pixel native parity."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
from mc_photogrammetry.reconstruction import region_growth as growth
import test_diagnose_image_proposal_attrition_v2 as inherited
from test_seed_gate_attribution import subject


def synthetic_hooks(observer,frames=("a","b")):
    """Explicit scalar source hooks only for stubbed-G serialization tests."""
    config=growth.DEFAULT_REGION_GROWTH_CONFIG
    best=np.full(3,.7,dtype=np.float32);other=np.full(3,.1,dtype=np.float32)
    support=np.full(3,20.,dtype=np.float32);sides=np.ones(3,dtype=np.int8)
    for frame in frames:
        observer.begin_frame(frame)
        observer.expect("frame",(frame,"proposal"));observer.finish("frame",(frame,"proposal"),"synthetic")
        for axis in range(3):
            for side in (-1,1):
                identity=(frame,axis,side);seed=sides==side
                observer.expect("seed",identity)
                observer.finish("seed",identity,"available" if side==1 else "absent",axis=axis,side=side,eligible_seed_count=3 if side==1 else 0)
                observer.seed_predicates(identity,count=3,best=best,other=other,seed_support=support,sides=sides,seed=seed,config=config)
            identity=(frame,axis)
            observer.expect("proposal",identity);observer.finish("proposal",identity,"returned",axis=axis,returned_ordinal=axis)
            observer.bind(("proposal_return",frame,axis),identity)
            observer.expect("reading",identity);observer.finish("reading",identity,"accepted",axis=axis)
        for prefix in ("proposal_generation","proposal_returned","proposal_to_reading"):observer.check(prefix+":"+frame,True)
        observer.end_frame(frame)


@pytest.fixture(autouse=True)
def scoped_subject(monkeypatch):
    monkeypatch.setattr(inherited,"subject",subject)
    monkeypatch.setattr(inherited,"scalar_events",synthetic_hooks)


def test_actual_measure_source_hook_and_strict_readback(tmp_path,monkeypatch):
    inherited.test_real_measure_frame_context_accounting_and_g7_identity(tmp_path,monkeypatch)
    report=json.loads((tmp_path/"attrition.json").read_text())
    seed=report["seed_attribution"]
    assert len(seed["rows"])==12 and all(r["state"]=="reached" for r in seed["rows"])
    assert all(sum(r["histogram"])==r["denominator"] for r in seed["rows"])


def test_actual_default_driver_off_on_output_digest_kernel_parity(tmp_path,monkeypatch):
    stages=("_local_lab_xy_atoms","_segment_support","_normal_scores","_build_graph","_seed_and_grow")
    calls={name:0 for name in stages}
    seed_inputs=[]
    for name in stages:
        original=getattr(growth,name)
        def counted(*args,_name=name,_original=original,**kwargs):
            calls[_name]+=1
            if _name=="_seed_and_grow":
                graph=args[0]
                before=(graph.plane_scores.tobytes(),graph.supports.tobytes(),graph.sides.tobytes())
                result=_original(*args,**kwargs)
                assert before==(graph.plane_scores.tobytes(),graph.supports.tobytes(),graph.sides.tobytes())
                seed_inputs.append((args[1],args[2],before,result))
                return result
            return _original(*args,**kwargs)
        monkeypatch.setattr(growth,name,counted)
    inherited.test_real_default_driver_off_on_pixel_universe_and_kernel_parity(tmp_path,monkeypatch)
    assert calls=={name:24 if name=="_seed_and_grow" else 4 for name in stages}
    assert seed_inputs[:12]==seed_inputs[12:]


def test_actual_unique_compact_success_calls_one_matcher_selector_evaluator(tmp_path,monkeypatch):
    inherited.test_genuine_unique_g7_is_compact_no_failure_without_invented_proof(tmp_path,monkeypatch)
    report=json.loads((tmp_path/"attrition.json").read_text())
    assert report["g7"]["proof_scope"]=="compact_no_failure"
    assert len(report["seed_attribution"]["rows"])==12


@pytest.mark.parametrize("failure",["short_write","fsync","link","race","temporary_readback","final_readback"])
def test_publication_failure_preserves_outputs_and_racer(failure,tmp_path,monkeypatch):
    inherited.test_atomic_failure_preserves_production_and_foreign_racer(failure,tmp_path,monkeypatch)


def test_publication_success_after_fsync_exact_readback_and_collision(tmp_path,monkeypatch):
    inherited.test_success_publication_not_visible_until_fsync_and_exact_readback(tmp_path,monkeypatch)


def test_delivery_context_native_error_and_created_artifacts(tmp_path):
    inherited.test_native_delivery_error_retains_g7_and_created_output(tmp_path)


def test_source_mismatch_prevents_delivery(tmp_path,monkeypatch):
    inherited.test_runtime_source_mismatch_stops_before_delivery(tmp_path,monkeypatch)


@pytest.mark.parametrize("mutation",["remove","duplicate","order","histogram","denominator","counts","eligible","state","not_reached","threshold","resources","limits","precedence","omit_field","seed_join","source_check","pair_stage","boolean_count","boolean_resource","finite_threshold_score","finite_threshold_margin","finite_threshold_support"])
def test_resealed_seed_binding_closure_and_accounting_refuses(mutation,tmp_path,monkeypatch):
    outcome,_,_=inherited.run(tmp_path,monkeypatch)
    assert outcome.diagnostic_error is None
    report=copy.deepcopy(outcome.report);record=report["seed_attribution"];rows=record["rows"];row=rows[0]
    if mutation=="remove":rows.pop()
    elif mutation=="duplicate":rows[1]=copy.deepcopy(row)
    elif mutation=="order":rows.reverse()
    elif mutation=="histogram":row["histogram"][0]+=1
    elif mutation=="denominator":row["denominator"]+=1
    elif mutation=="counts":row["counts"]["passes"][0]+=1
    elif mutation=="eligible":row["eligible_seed_count"]+=1
    elif mutation=="state":row["state"]="pending"
    elif mutation=="not_reached":
        row.update(state="not_reached",source_disposition="not_reached",reason="forged",denominator=None,histogram=None,counts=None,eligible_seed_count=None,thresholds=None)
    elif mutation=="threshold":row["thresholds"]=[True,0.,12.]
    elif mutation.startswith("finite_threshold_"):
        index={"finite_threshold_score":0,"finite_threshold_margin":1,"finite_threshold_support":2}[mutation]
        row["thresholds"][index]=float(np.nextafter(row["thresholds"][index],np.inf))
        assert row["thresholds"][index]!=subject.SEED_THRESHOLDS[index]
        assert np.isfinite(row["thresholds"]).all()
    elif mutation=="resources":record["resources"]["atom_visits"]+=1
    elif mutation=="limits":record["limits"]["atom_visits"]+=1
    elif mutation=="precedence":record["precedence"].reverse()
    elif mutation=="omit_field":del row["source_disposition"]
    elif mutation=="seed_join":
        # Exact local equations still close; independently bound source metric differs.
        row["histogram"]=[0]*16;row["histogram"][15]=3
        row["counts"]=subject.seed_counts(row["histogram"]);row["eligible_seed_count"]=3;row["source_disposition"]="available"
    elif mutation=="source_check":del report["observer"]["checks"]["signature_universe"]
    elif mutation=="pair_stage":del report["observer"]["stages"]["pair"]
    elif mutation=="boolean_count":row["counts"]["first_failures"][0]=False
    elif mutation=="boolean_resource":record["resources"]["frames"]=True
    with pytest.raises(subject.AttritionError):subject.validate_report(subject._seal(report),outcome.g7_outcome.report)


def test_incomplete_seed_callback_preserves_production_and_g7(tmp_path,monkeypatch):
    monkeypatch.setattr(subject.StreamingObserver,"seed_predicates",lambda *a,**k:None)
    outcome,graph,_=inherited.run(tmp_path,monkeypatch)
    assert outcome.g7_outcome.production_result is graph
    assert outcome.diagnostic_error.reason=="observer_incomplete"
    assert not (tmp_path/"attrition.json").exists() and (tmp_path/"g7.json").exists() and (tmp_path/"out.mcscan").exists()


def test_escaped_byte_and_lexical_limits_before_decoder_or_encoder(monkeypatch):
    inherited.test_sidecar_exact_four_mib_and_one_over_precharge(monkeypatch)


@pytest.mark.parametrize("mutation",["denominator","joint_marginal","overlapping_sides"])
def test_resealed_locally_valid_rows_violate_shared_graph_joins(mutation,tmp_path,monkeypatch):
    outcome,_,_=inherited.run(tmp_path,monkeypatch)
    assert outcome.diagnostic_error is None
    report=copy.deepcopy(outcome.report);record=report["seed_attribution"]
    row=next(row for row in record["rows"] if (row["frame"],row["axis"],row["side"])==("a",0,-1))
    if mutation=="denominator":
        row["histogram"][7]+=1;row["denominator"]+=1
        record["resources"]["atom_visits"]+=1;record["resources"]["predicate_visits"]+=4
    elif mutation=="joint_marginal":
        row["histogram"][7]-=1;row["histogram"][6]+=1
    else:
        # First-three joint marginals remain equal, but both side predicates
        # claiming the same atoms violates source side-sign exclusivity.
        row["histogram"][7]=0;row["histogram"][15]=3
        row["eligible_seed_count"]=3;row["source_disposition"]="available"
        stage=report["observer"]["stages"]["seed"]
        absent=next(x for x in stage["rows"] if x["frame"]=="a" and x["axis"]==0 and x["disposition"]=="absent")
        stage["rows"].remove(absent)
        available=next(x for x in stage["rows"] if x["frame"]=="a" and x["axis"]==0 and x["disposition"]=="available")
        available["count"]=2;available["eligible_seed_count"].update(count=2,total=6)
        report["observer"]["resources"]["aggregate_buckets"]-=1
    row["counts"]=subject.seed_counts(row["histogram"])
    assert sum(row["histogram"])==row["denominator"]
    assert row["counts"]["all_pass"]==row["eligible_seed_count"]
    assert sum(row["counts"]["first_failures"])+row["counts"]["all_pass"]==row["denominator"]
    with pytest.raises(subject.AttritionError,match="shared_frame_denominator|paired_side_mask_join"):
        subject.validate_report(subject._seal(report),outcome.g7_outcome.report)


def test_resealed_locally_closed_axis_pair_cannot_change_frame_reachability(tmp_path,monkeypatch):
    outcome,_,_=inherited.run(tmp_path,monkeypatch)
    assert outcome.diagnostic_error is None
    report=copy.deepcopy(outcome.report);record=report["seed_attribution"]
    for row in record["rows"]:
        if row["frame"]=="a" and row["axis"]==0:
            row.update(state="not_reached",source_disposition="not_reached",reason="synthetic",
                denominator=None,histogram=None,counts=None,eligible_seed_count=None,thresholds=None)
    record["resources"]["atom_visits"]-=6
    record["resources"]["predicate_visits"]-=24
    stage=report["observer"]["stages"]["seed"]
    replaced=[row for row in stage["rows"] if row["frame"]=="a" and row["axis"]==0]
    assert len(replaced)==2
    replacement=copy.deepcopy(replaced[0])
    replacement.update(disposition="not_reached",count=2,eligible_seed_count=None)
    stage["rows"]=[row for row in stage["rows"] if row not in replaced]+[replacement]
    stage["rows"].sort(key=lambda row:(row["frame"],row["axis"],row["disposition"]))
    report["observer"]["resources"]["aggregate_buckets"]-=1
    # Both local sides and the aggregate seed count still close. The other
    # four identities nevertheless came from the same reached source graph.
    assert len([row for row in record["rows"] if row["frame"]=="a" and row["state"]=="not_reached"])==2
    with pytest.raises(subject.AttritionError,match="shared_frame_reachability"):
        subject.validate_seed_record(record,report["observer"],{"a","b"})
    # The full report also binds seed availability to returned proposals and
    # rejects this reseal at that earlier independent source join.
    with pytest.raises(subject.AttritionError):
        subject.validate_report(subject._seal(report),outcome.g7_outcome.report)
