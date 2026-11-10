"""Read-only independent population audit and supplemental G4 diagnostics."""
from collections import Counter, defaultdict
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.distance import pdist

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.ai import geometry19_performance_gated_g4 as m


def quantiles(values):
    return dict(count=len(values), p50=None if not values else float(np.quantile(values,.5)),
                p95=None if not values else float(np.quantile(values,.95)), maximum=None if not values else max(values))


def main():
    config,data,base=m.preflight(ROOT)
    output=ROOT/config['output_root']
    lock=json.loads((output/'Protocol_Lock.json').read_text())
    summary=m.verify_complete(output,lock)
    destination=output.parent/'g4_rank128_v1_audit'
    destination.mkdir(exist_ok=False)
    physics=m.read_rows(output/'Physical_Evaluations.jsonl')
    slots=m.read_rows(output/'Selection_Rank128.jsonl')
    ledger=m.read_rows(output/'Ranked_Physics_Ledger.jsonl')
    tasks=m.read_rows(output/'Task_Metrics.jsonl')
    tolerances=data['parent']['evaluation']['relative_tolerances']
    groups=defaultdict(list); cross=Counter(); coverage=Counter()
    used=set(); support_violations=integer_violations=0; detail=[]
    for s,r in zip(slots,ledger,strict=True):
        assert all(s[k]==r[k] for k in s if k!='geometry19')
        if s['missing']:
            assert not r['hit'] and r['rho'] is None and r['loss']==1.
            continue
        p=physics[s['physics_index']]; used.add(s['physics_index'])
        assert p['design_hash']==s['design_hash'] and p['geometry19']==s['geometry19']
        assert p['pipeline_context_hash']==data['context'].context_hash
        assert p['physics_bundle_hash']==data['context'].physics_runtime.physics_bundle_hash
        values=p.get('performance') or {}
        finite=all(isinstance(values.get(k),(int,float)) and math.isfinite(values[k]) and values[k]>0 for k in m.NAMES)
        target=data['targets'][s['target_index']]
        eligible=(p['solver']['converged'] and p['constraint_assessment']['numerically_valid'] and
                  p['constraint_assessment']['physics_domain_valid'] and
                  (p.get('target_domain_assessment') or {}).get('inside') is True and finite)
        rho=max(abs(values[k]/target['performance'][k]-1)/tolerances[k] for k in m.NAMES) if finite else None
        assert (rho is None and r['rho'] is None) or math.isclose(rho,r['rho'],rel_tol=1e-10,abs_tol=1e-12)
        assert r['physics_eligible']==eligible and r['hit']==bool(eligible and rho<=1.)
        assert math.isclose(r['loss'],min(rho,10)/10 if eligible else 1.,rel_tol=1e-10,abs_tol=1e-12)
    assert len(used)==len(physics)==summary['actual_calls']
    for i,task in enumerate(m.g3.task_list(range(99))):
        seed,index,arm=task
        part=ledger[i*128:(i+1)*128]; selected=slots[i*128:(i+1)*128]
        assert [r['selection_rank'] for r in part]==list(range(1,129))
        assert all((r['training_seed'],r['target_index'],r['arm'])==task for r in part)
        target=data['targets'][index]
        assert m.task_metrics(part,target)==tasks[i]
        raw=m.read_rows(base/'raw'/m.g3.task_key(task)/'Raw.jsonl')
        prediction=m.read_rows(base/'ranked'/m.g3.task_key(task)/'Predictions.jsonl')
        old=m.read_rows(base/'ranked'/m.g3.task_key(task)/'Selection.jsonl')
        assert [{k:v for k,v in s.items() if k!='physics_index'} for s in selected]==m.task_selection(raw,prediction,old)
        meta=json.loads((base/'raw'/m.g3.task_key(task)/'Task.json').read_text())
        valid=[r for r in raw if r['valid']]
        coordinates=np.asarray([[r['geometry19'][k] for k in data['decoders']['continuous_axes']] for r in valid])
        distances=pdist(coordinates/np.asarray(data['decoders']['continuous_span'])) if len(valid)>1 else []
        detail.append(dict(seed=seed,target_index=index,arm=arm,source_root_id=target['root_seed_id'],
                           raw_valid=len(valid),raw_valid_fraction=len(valid)/256,
                           valid_duplicate_count=len(valid)-len({r['design_hash'] for r in valid}),
                           schema_normalized_16d_pair_distances=quantiles(list(distances)),
                           generation_seconds=meta['elapsed_seconds']))
        for s,r in zip(selected,part):
            if s['missing']: continue
            rawrow=raw[s['raw_slot_id']]
            support_violations+=int(not rawrow['support_valid'] or not rawrow['geometry_hard_valid'])
            expert=data['decoders']['experts'][rawrow['support_root_index']]
            integers=[s['geometry19'][axis] for axis in data['decoders']['integer_axes']]
            integer_violations+=int(integers!=expert['integer_values'] or any(int(v)!=v for v in integers))
            same_generator=(None if rawrow['selected_expert_index'] is None else data['decoders']['root_order'][rawrow['selected_expert_index']]==target['root_seed_id'])
            if s['selection_rank']==1:
                cross[(seed,arm,str(same_generator),str(r['hit']))]+=1
            for k in (1,8,128):
                if s['selection_rank']<=k:
                    coverage[(arm,k,'same_support_root' if s['support_root_id']==target['root_seed_id'] else 'other_support_root','hit' if r['hit'] else 'miss')]+=1
        for k in (1,8,128):
            for group in [f'root/{target["root_seed_id"]}',f'class/{target["target_class"]}',*[f'segment/{s}' for s in target['source_root_segments']]]:
                groups[(arm,k,group)].extend(part[:k])
    assert support_violations==integer_violations==0
    assert [r for r in ledger if r['selection_rank']<=8]==m.read_rows(output/'Original_G4_Top8_Ledger.jsonl')
    group_rows=[]
    for (arm,k,group),rows in sorted(groups.items()):
        counts={name:sum(bool(r[name]) for r in rows) for name in ('missing','solver_converged','numerically_valid','physics_domain_valid','target_domain_inside','physics_eligible','hit')}
        group_rows.append(dict(arm=arm,rank_cutoff=k,group=group,slots=len(rows),counts=counts,
                               eligible_rho=quantiles([r['rho'] for r in rows if r['physics_eligible']]),
                               fixed_denominator_loss=quantiles([r['loss'] for r in rows])))
    m.write_rows(destination/'Group_Diagnostics.jsonl',group_rows)
    m.write_rows(destination/'Generation_Diversity.jsonl',detail)
    m.write_rows(destination/'Gate_Rank1_Hit_Cross_Table.jsonl',[dict(seed=s,arm=a,generator_is_source_root=g,hit=h,tasks=n) for (s,a,g,h),n in sorted(cross.items())])
    m.write_rows(destination/'Support_Root_Hit_Cross_Table.jsonl',[dict(arm=a,rank_cutoff=k,support_relation=g,result=h,slots=n) for (a,k,g,h),n in sorted(coverage.items())])
    root_summary=[]
    for arm in m.g3.ARMS:
        for root in data['decoders']['root_order']:
            sub=[t for t in tasks if t['arm']==arm and t['source_root_id']==root]
            root_summary.append(dict(arm=arm,root_seed_id=root,**m.aggregate(sub)))
    m.write_rows(destination/'Per_Root_Rank_Curves.jsonl',root_summary)
    audit=dict(status='G4_INDEPENDENT_POPULATION_AUDIT_PASS',slots=len(slots),tasks=len(tasks),actual_calls=len(physics),
               original_top8_exact=True,all_slot_hits_independently_recomputed=True,
               selected_support_violations=support_violations,selected_integer_violations=integer_violations,
               first_hit_histogram={a:dict(Counter(str(t['first_hit_rank']) for t in tasks if t['arm']==a)) for a in m.g3.ARMS},
               unhit_tasks=[{k:t[k] for k in ('training_seed','target_index','arm','source_root_id','target_class')} for t in tasks if t['first_hit_rank'] is None],
               physics_worker_seconds=sum(s['elapsed_seconds'] for s in json.loads((output/'Acquisition_Manifest.json').read_text())['seals']),
               generation_seconds=sum(d['generation_seconds'] for d in detail),
               final_complete_sha256=m.file_hash(output/'Complete_Manifest.json'),
               note='Rho quantiles use eligible candidates only; missing/invalid counts and fixed-denominator loss are reported separately. Segment groups may overlap.')
    m.g3.seal_json(destination/'Audit_Summary.json',audit)
    m.g3.seal_json(destination/'Audit_Manifest.json',dict(script_sha256=m.file_hash(Path(__file__)),
                  files={p.name:m.file_hash(p) for p in destination.iterdir() if p.is_file()}))
    print(json.dumps({k:audit[k] for k in ('status','slots','tasks','actual_calls','selected_support_violations','selected_integer_violations')},indent=2))


if __name__=='__main__':
    main()
