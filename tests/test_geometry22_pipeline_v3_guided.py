import json
from pathlib import Path
import pytest
from src.ai import geometry22_pipeline_v3_guided as p
from src.ai import geometry22_pipeline_v3_cfm as cfm
from src.ai.geometry19_pipeline_artifact import verify_completed_artifact
ROOT=Path(__file__).resolve().parents[1]

CHECK=ROOT / 'data/validation/v3_22d_fresh_validation_l6h_v1'


@pytest.mark.parametrize('rows,expected', [
    ([], None),
    ([('unscored', None, False, True)], None),
    ([('invalid', .01, False, False), ('valid', .1, False, True)], 'valid'),
    ([('valid', .01, False, True), ('feasible', .9, True, True)], 'feasible'),
    ([('invalid_b', .1, False, False), ('invalid_a', .1, False, False)], 'invalid_a'),
    ([('valid_b', .1, False, True), ('valid_a', .1, False, True)], 'valid_a'),
])
def test_f_selection_priority_and_empty_fallback(rows, expected):
    records = [dict(design_id=name, rho=rho, feasible=feasible, diagnostic_domain_valid=valid)
               for name, rho, feasible, valid in rows]
    best = p.best_result(records)
    assert (best['design_id'] if best else None) == expected


def test_archived_f_selection_retains_legacy_policy():
    records = [dict(design_id='invalid', rho=.01, feasible=False, diagnostic_domain_valid=False),
               dict(design_id='valid', rho=.1, feasible=False, diagnostic_domain_valid=True)]
    assert p.best_result(records, p.LEGACY_F_SELECTION_POLICY)['design_id'] == 'invalid'
    with pytest.raises(ValueError, match='Unknown guided F selection policy'):
        p.best_result(records, 'unknown')

@pytest.fixture
def fake_flow(monkeypatch):
    target = json.loads((CHECK / "target_labels.jsonl").read_text().splitlines()[0])["target_performance"]
    raw = [json.loads(s) for s in (CHECK / "raw_candidates.jsonl").read_text().splitlines()]
    original_id = raw[0]["target_id"]
    raw = [r for r in raw if r["target_id"] == original_id and r["arm"] == "gated_top2"]
    def generate(root, requested, cfg, local, support, norm, checkpoint):
        rows = [{**r, "target_id": requested["target_id"], "generator":"v3_local_cfm_v1", "generator_checkpoint_sha256":cfg["sources"]["cfm_checkpoint_sha256"]} for r in raw]
        ranked = sorted(rows, key=lambda r: (r["predicted_rho"], r["design_id"], r["raw_slot"]))[:24]
        selected = [{**r, "rank": i} for i, r in enumerate(ranked)]
        route = {"target_id": requested["target_id"], "status": "routable", "eligible_center_count": 2,
                 "arms": {"gated_top2": sorted({r["route_center_id"] for r in rows})}}
        return selected, rows, route
    def evaluate(row, runtime, profile, requested, cfg):
        return {"design_id": row["design_id"], "design": row["design"], "route_center_id": row["route_center_id"],
                "design_schema_hash": runtime.design_schema_hash, "physics_bundle_hash": runtime.physics_bundle_hash,
                "solver_converged": True, "solver_failure_code": None, "diagnostic_domain_valid": True,
                "feasible": False, "performance": target, "rho": cfm._rho(target, requested, cfg["relative_tolerances"]),
                "software_hit_5pct": True, "formal_use_allowed": False, "deployment_allowed": False}
    monkeypatch.setattr(cfm, "generate", generate)
    monkeypatch.setattr(cfm, "evaluate_design", evaluate)
    return target, evaluate


@pytest.fixture
def flow(fake_flow,monkeypatch):
    target,evaluate=fake_flow
    monkeypatch.setattr(p,'generate',cfm.generate)
    monkeypatch.setattr(p,'evaluate_design',evaluate)
    return target,evaluate


def test_natural_c_hit_skips_d_and_e_and_resumes_without_generation(flow,monkeypatch,tmp_path):
    target,_=flow
    monkeypatch.setattr(p,'refine_candidates',lambda *_:pytest.fail('D must skip C hit'))
    out=tmp_path/'hit';result=p.run_v3_routed_pipeline(ROOT,target_values=target,output_root=out)
    assert result['physics_evaluation_count']==9
    assert p.read(out/'D/candidates.json')==p.read(out/'E/candidates.json')==[]
    assert verify_completed_artifact(out,ROOT)['integrity_pass']
    monkeypatch.setattr(p,'generate',lambda *_:pytest.fail('sealed B must not rerun'))
    assert p.run_v3_routed_pipeline(ROOT,target_values=target,output_root=out,resume=True)==result


def test_c_valid_hit_stops_search_and_f_ignores_invalid_lower_error(flow, monkeypatch, tmp_path):
    target, evaluate = flow
    invalid_id = None
    def mixed(row, *args):
        nonlocal invalid_id
        if invalid_id is None:
            invalid_id = row['design_id']
        invalid = row['design_id'] == invalid_id
        result = evaluate(row, *args)
        result['performance'] = {k: v * (1.001 if invalid else 1.01) for k, v in target.items()}
        result['rho'] = p._rho(result['performance'], target, args[-1]['relative_tolerances'])
        result['diagnostic_domain_valid'] = not invalid
        return result
    monkeypatch.setattr(p, 'evaluate_design', mixed)
    monkeypatch.setattr(p, 'refine_candidates', lambda *_: pytest.fail('valid C hit must skip D'))
    out = tmp_path / 'mixed'
    result = p.run_v3_routed_pipeline(ROOT, target_values=target, output_root=out)
    assert result['physics_evaluation_count'] == 9
    assert p.read(out/'D/candidates.json') == p.read(out/'E/candidates.json') == []
    evaluated = p.read(out/'C/evaluations.json')
    final = p.read(out/'F/summary.json')
    best = p.best_result(evaluated)
    assert final['selected_design_id'] == best['design_id'] != invalid_id
    assert final['independent_recompute']['diagnostic_domain_valid']
    assert final['software_hit_5pct'] and final['status'] == 'NO_HIT'
    assert p.read(out/'Protocol_Lock.json')['identity']['protocol']['f_recompute'] == p.F_SELECTION_POLICY
    assert verify_completed_artifact(out, ROOT)['integrity_pass']
    assert p.run_v3_routed_pipeline(ROOT, target_values=target, output_root=out, resume=True) == result


def test_c_miss_d_updates_then_hit_skips_e_and_f_recomputes(flow,monkeypatch,tmp_path):
    target,evaluate=flow
    def refined(seeds,*args):
        return [{**r,'parent_design_id':r['design_id']} for r in seeds], [{'component_probe':'synthetic stage trigger'}]
    monkeypatch.setattr(p,'refine_candidates',refined)
    calls=0
    def assess(row,*args):
        nonlocal calls
        calls+=1
        result=evaluate(row,*args)
        if calls<=8:
            result['performance']={k:v*1.2 for k,v in target.items()};result['rho']=p._rho(result['performance'],target,args[-1]['relative_tolerances']);result['software_hit_5pct']=False
        return result
    monkeypatch.setattr(p,'evaluate_design',assess)
    out=tmp_path/'rescue';result=p.run_v3_routed_pipeline(ROOT,target_values=target,output_root=out)
    assert result['physics_evaluation_count']==17 and result['software_hit_5pct']
    assert len(p.read(out/'D/candidates.json'))==8 and p.read(out/'E/candidates.json')==[]
    assert verify_completed_artifact(out,ROOT)['integrity_pass']


def test_no_hit_reaches_e_and_abstention_calls_no_refinement(flow,monkeypatch,tmp_path):
    target,evaluate=flow
    monkeypatch.setattr(p,'refine_candidates',lambda seeds,*_:([{**r,'parent_design_id':r['design_id']} for r in seeds],[]))
    def miss(row,*args):
        result=evaluate(row,*args);result['performance']={k:v*1.2 for k,v in target.items()};result['rho']=p._rho(result['performance'],target,args[-1]['relative_tolerances']);result['software_hit_5pct']=False;return result
    monkeypatch.setattr(p,'evaluate_design',miss)
    result=p.run_v3_routed_pipeline(ROOT,target_values=target,output_root=tmp_path/'miss')
    assert result['physics_evaluation_count']==25 and not result['software_hit_5pct']
    monkeypatch.setattr(p,'generate',lambda root,t,*a:([],[],{'target_id':t['target_id'],'status':'abstained_no_target_support','eligible_center_count':0,'arms':{'gated_top2':[]}}))
    monkeypatch.setattr(p,'refine_candidates',lambda *a:pytest.fail('empty seeds must not refine'))
    result=p.run_v3_routed_pipeline(ROOT,target_values=target,output_root=tmp_path/'abstain')
    assert result['physics_evaluation_count']==0


def test_resume_after_c_miss_reuses_c_and_runs_d(flow,monkeypatch,tmp_path):
    target,evaluate=flow;calls=0
    monkeypatch.setattr(p,'refine_candidates',lambda seeds,*_:([{**r,'parent_design_id':r['design_id']} for r in seeds],[]))
    def interrupt(row,*args):
        nonlocal calls
        calls+=1
        if calls==9:raise KeyboardInterrupt()
        result=evaluate(row,*args)
        result['performance']={k:v*1.2 for k,v in target.items()};result['rho']=p._rho(result['performance'],target,args[-1]['relative_tolerances']);result['software_hit_5pct']=False;return result
    monkeypatch.setattr(p,'evaluate_design',interrupt)
    out=tmp_path/'interrupted'
    with pytest.raises(KeyboardInterrupt):p.run_v3_routed_pipeline(ROOT,target_values=target,output_root=out)
    assert p.read(out/'Run_State.json')['completed_stages']==['B','C']
    before=p.sha(out/'C/evaluations.json')
    monkeypatch.setattr(p,'generate',lambda *_:pytest.fail('B must not regenerate'))
    monkeypatch.setattr(p,'evaluate_design',evaluate)
    result=p.run_v3_routed_pipeline(ROOT,target_values=target,output_root=out,resume=True)
    assert p.sha(out/'C/evaluations.json')==before and result['physics_evaluation_count']==17
    assert verify_completed_artifact(out,ROOT)['integrity_pass']
