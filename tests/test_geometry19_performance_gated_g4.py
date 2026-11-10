from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.ai import geometry19_performance_gated_g4 as m


def slot(rank=1, missing=False):
    return dict(training_seed=42,target_index=0,target_id='target',generation_seed=20261001,
                arm=m.g3.ARMS[0],selection_rank=rank,missing=missing,geometry19={'x':1},
                design_hash='geometry',physics_index=0)


def physics():
    return dict(design_hash='geometry',solver={'converged':True},
                constraint_assessment={'numerically_valid':True,'physics_domain_valid':True},
                target_domain_assessment={'inside':True},performance={k:100. for k in m.NAMES})


TARGET=dict(performance={k:100. for k in m.NAMES},root_seed_id='root',target_class='outer')
TOL={k:.1 for k in m.NAMES}


def test_hit_boundary_and_relative_tolerance():
    p=physics(); p['performance']['thrust_N']=110.
    result=m.score_slot(slot(),p,TARGET,TOL)
    assert result['rho']==1. and result['hit'] and result['loss']==.1
    p['performance']['thrust_N']=110.001
    assert not m.score_slot(slot(),p,TARGET,TOL)['hit']


@pytest.mark.parametrize('failure',['solver','numerical','domain','target','nonfinite','negative','missing'])
def test_all_eligibility_failures_stay_in_denominator(failure):
    p=physics(); s=slot()
    if failure=='solver': p['solver']['converged']=False
    if failure=='numerical': p['constraint_assessment']['numerically_valid']=False
    if failure=='domain': p['constraint_assessment']['physics_domain_valid']=False
    if failure=='target': p['target_domain_assessment']=None
    if failure=='nonfinite': p['performance']['mass_kg']=float('nan')
    if failure=='negative': p['performance']['mass_kg']=-1.
    if failure=='missing': s['missing']=True; p=None
    result=m.score_slot(s,p,TARGET,TOL)
    assert not result['hit'] and result['loss']==1.


def test_first_hit_is_rank_not_best_physics_and_no_early_stop():
    rows=[m.score_slot(slot(k),physics(),TARGET,TOL) for k in range(1,129)]
    for row in rows[:63]: row['hit']=False
    result=m.task_metrics(rows,TARGET)
    assert result['first_hit_rank']==64 and result['hit_count_128']==65
    assert result['cumulative_hit_count'][7]==0 and result['at_least_half_hit']
    combined=m.aggregate([result])
    assert combined['hit_at_k'][62]==0 and combined['hit_at_k'][63]==1
    assert combined['candidate_hit_fraction_128']==65/128


def test_score_cache_is_target_independent():
    p=physics()
    different=deepcopy(TARGET); different['performance']['mass_kg']=200.
    assert m.score_slot(slot(),p,TARGET,TOL)['hit']
    assert not m.score_slot(slot(),p,different,TOL)['hit']


def fixture_task():
    raw=[]; pred=[]; old=[]
    for i in range(256):
        row=dict(target_id='target',target_index=0,arm=m.g3.ARMS[0],training_seed=42,generation_seed=20261001,
                 raw_slot_id=i,geometry19={'x':i},design_hash=f'{i:03d}',support_root_id='root',valid=True,
                 support_valid=True,geometry_hard_valid=True)
        raw.append(row)
        pred.append(dict(row,rank=i+1,predicted_rho=float(i)))
        if i<8:
            old.append({k:row[k] for k in ('target_id','target_index','arm','training_seed','generation_seed',
                                         'raw_slot_id','geometry19','design_hash','support_root_id')})
            old[-1].update(selection_rank=i+1,missing=False,predicted_rho=float(i))
    return raw,pred,old


def test_rank128_preserves_exact_top8_and_no_reranking():
    raw,pred,old=fixture_task()
    selected=m.task_selection(raw,pred,old)
    assert selected[:8]==old and len(selected)==128 and selected[-1]['raw_slot_id']==127
    pred[0]['rank'],pred[1]['rank']=2,1
    with pytest.raises(Exception,match='ordering'): m.task_selection(raw,pred,old)


def test_insufficient_ranked_population_is_not_topped_up():
    raw,pred,old=fixture_task()
    for p in pred[8:]: p['rank']=None; p['valid']=False
    selected=m.task_selection(raw,pred,old)
    assert sum(s['missing'] for s in selected)==120
    assert all(s['geometry19'] is None for s in selected[8:])


def test_physics_journal_resume_tamper_and_orphan(tmp_path,monkeypatch):
    (tmp_path/'physics').mkdir()
    batch=[dict(physics_index=0,design_hash='geometry',geometry19={'x':1})]
    context=SimpleNamespace(context_hash='ctx')
    monkeypatch.setattr(m,'CONTEXT',context)
    calls=[]
    def evaluator(row,ctx):
        calls.append(row)
        return dict(row,pipeline_context_hash=ctx.context_hash)
    monkeypatch.setattr(m,'_evaluate_row',evaluator)
    _,seal=m.acquire_job((str(tmp_path),0,batch,'lock'))
    m.acquire_job((str(tmp_path),0,batch,'lock'))
    assert len(calls)==1 and seal['actual_calls']==1
    path,journal,sealp=m.shard_paths(tmp_path,0)
    path.write_text('{}\n')
    with pytest.raises(Exception,match='hash'): m.verify_shard(tmp_path,0,batch,'lock')
    sealp.unlink()
    with pytest.raises(Exception,match='Orphan'): m.verify_shard(tmp_path,0,batch,'lock')


def test_actual_frozen_g3_contract_without_physics(monkeypatch):
    from src.physics.runtime import PhysicsRuntime
    monkeypatch.setattr(PhysicsRuntime,'evaluate',lambda *a,**kw:pytest.fail('preflight evaluated physics'))
    root=Path(__file__).resolve().parents[1]
    config,data,base=m.preflight(root)
    assert data['context'].physics_runtime.model_version=='geometry19_v2_2_far_precision_v1'
    assert config['population']['slots']==152064
