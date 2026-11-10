"""N3 adapter: reuse WP19-23 numerical engines with local models and V2.2.

This is an explicit experimental entry, not a replacement for the production
B--F artifact/UI entry. G3/G4 frozen modules and physics identity stay intact.
"""
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch
from torch import nn
import yaml

from src.ai import geometry19_performance_gated_g3 as g3
from src.ai import geometry19_nsga2 as nsga
from src.ai.geometry19_candidates import _dflow_generate
from src.ai.geometry19_cfm import integrate_cfm_sources
from src.ai.geometry19_delivery import performance_matches
from src.ai.geometry19_forward_model import ForwardModelSpec
from src.ai.geometry19_inverse_contract import Geometry19InverseTarget, DiagnosticAcceptance
from src.ai.geometry19_physics import evaluate_geometry19_candidate, _json_safe
from src.ai.geometry19_pipeline import build_target_mapping
from src.ai.geometry19_refinement import RefinementModels, _surrogate_objective
from src.ai.geometry19_performance_gated_protocol import canonical, file_hash, require
from src.physics import Environment

CONFIG='config/experiments/geometry19_local_full_pipeline_smoke_v1.yaml'


class LocalFlow(nn.Module):
    """CFMGenerator numerical interface; integrate local 16D and decode on exit."""
    design_dim=16
    condition_dim=3
    model_config={'ode_steps':32,'ode_method':'rk4'}

    def __init__(self, model, decoders, root_index):
        super().__init__()
        self.model=model
        self.root_index=root_index
        expert=decoders['experts'][root_index]
        self.register_buffer('center',torch.tensor(expert['normalized_medoid_continuous'],dtype=torch.float32))
        self.radius=expert['radius']
        self.normalizer=SimpleNamespace(design=[dict(physical_min=l,physical_max=l+s) for l,s in zip(decoders['continuous_lower'],decoders['continuous_span'])])

    def velocity(self,t,state,condition):
        return self.model(state,t.expand(len(state)),condition,torch.full((len(state),),self.root_index,device=state.device,dtype=torch.long))

    def _from_model_space(self,state):
        return 2*(self.center+self.radius*state)-1,None,None


class LocalRidge(nn.Module):
    """Exact frozen local Ridge in log performance, differentiable continuous axes."""
    def __init__(self,data,scorer_ids,scorer,root_index):
        super().__init__()
        dec=data['decoders']; expert=dec['experts'][root_index]
        fields=sorted(data['schema']['design_variables'],key=lambda f:f['index'])
        route=scorer_ids.index(expert['root_seed_id'])
        coef=np.asarray(scorer['coefs'][route]); intercept=np.asarray(scorer['intercepts'][route]).copy()
        names=[f['name'] for f in fields]
        for axis,value in zip(dec['integer_axes'],expert['integer_values']):
            i=names.index(axis); lo,hi=fields[i]['bounds']; intercept+=coef[:,i]*(value-lo)/(hi-lo)
        self.register_buffer('coef',torch.tensor(coef[:,[names.index(a) for a in dec['continuous_axes']]],dtype=torch.float64))
        self.register_buffer('intercept',torch.tensor(intercept,dtype=torch.float64))
        self.register_buffer('center',torch.tensor(expert['normalized_medoid_continuous'],dtype=torch.float64))
        self.support_radius=expert['radius']*.20

    def forward(self,x):
        safe=x.clamp(0,1)
        delta=safe-self.center
        factor=(self.support_radius/torch.linalg.vector_norm(delta,dim=-1,keepdim=True).clamp_min(1e-20)).clamp(max=1.)
        return (self.center+delta*factor)@self.coef.T+self.intercept


def project_local(normalized,center,radius):
    """Optimizer-only projection, never applied to raw generated candidates."""
    z=((normalized+1)/2).clamp(0,1)
    delta=z-center
    factor=(radius*.20/torch.linalg.vector_norm(delta,dim=-1,keepdim=True).clamp_min(1e-20)).clamp(max=1.)
    return 2*(center+delta*factor)-1


def execute_stages(c,d,e,f):
    """The mature conditional C -> D -> E -> independent F sequence."""
    visited=[]
    result=None
    for name,stage in (('C',c),('D',d),('E',e)):
        visited.append(name)
        result=stage()
        if result is not None and result['development_diagnostic_hit']:
            visited.append('F')
            return dict(visited=visited,first_hit_stage=name,delivery=f(result))
    return dict(visited=visited,first_hit_stage=None,delivery=None)


def evaluate_until_hit(candidates, evaluate, stop_on_first_hit=False):
    """Evaluate a population in order, optionally stopping at the first true hit."""
    results = []
    first_hit = None
    for candidate in candidates:
        value = evaluate(candidate)
        if value is None:
            continue
        results.append(value)
        if value['evaluation']['development_diagnostic_hit'] and first_hit is None:
            first_hit = value['evaluation']
            if stop_on_first_hit:
                break
    return results, first_hit


class LocalTask:
    def __init__(self,data,config,models,scorer_ids,scorer,index,output):
        self.data,self.config,self.output=data,config,output
        self.public=data['runtime_targets'][index]
        self.target=Geometry19InverseTarget.from_mapping(build_target_mapping(**self.public['performance']))
        self.context=data['context']; self.dec=data['decoders']; self.index=index
        self.tolerances=data['parent']['evaluation']['relative_tolerances']
        self.acceptance=DiagnosticAcceptance(self.tolerances,dict(zip(g3.adapter.OUTPUT_NAMES,[1e-12]*3)),False)
        condition=g3.transform_condition(self.public['performance'],data['transform'])
        self.condition=torch.tensor(condition,device='cuda:0',dtype=torch.float32).reshape(1,3)
        with torch.no_grad(): self.route=int(models['gate'](self.condition).argmax(dim=1).item())
        self.flow=LocalFlow(models['local_cfm'],self.dec,self.route).to('cuda:0')
        self.surrogate=LocalRidge(data,scorer_ids,scorer,self.route).to('cuda:0',dtype=torch.float32)
        self.ids,self.scorer=scorer_ids,scorer
        self.lower=torch.tensor(self.dec['continuous_lower'],device='cuda:0',dtype=torch.float32)
        self.span=torch.tensor(self.dec['continuous_span'],device='cuda:0',dtype=torch.float32)
        spec=ForwardModelSpec('local_ridge','frozen_S2_local_Ridge',data['paths']['s2_oracle_local_ridge'].parent,
                              'natural_log','performance3',tuple(g3.adapter.OUTPUT_NAMES),False,False)
        self.stats=dict(input_mean=np.asarray(self.dec['continuous_lower']),input_std=np.asarray(self.dec['continuous_span']),
                        output_mean=np.zeros(3),output_std=np.ones(3),_forward_model_spec=spec)
        self.ref_models=RefinementModels(self.surrogate,self.stats,spec,None,None)
        self.protocol=deepcopy(dict(self.context.protocol))
        self.protocol['diagnostic_acceptance']['relative_tolerances']=self.tolerances
        self.protocol['dflow']['steps']=config['guidance']['source_steps']
        self.cache={}; self.ledger=[]; self.costs=Counter(); self.physical_rows=[]
        output.mkdir(parents=True,exist_ok=False)

    def decode(self,normalized):
        state=((normalized.detach().cpu().numpy()+1)/2-np.asarray(self.dec['experts'][self.route]['normalized_medoid_continuous']))/self.flow.radius
        return g3.decode_rows(state,np.full(len(state),self.route),None,self.dec,self.data['schema'],self.context.physics_runtime.model_config)

    def normalized(self,geometry):
        x=torch.tensor([geometry[k] for k in self.dec['continuous_axes']],device='cuda:0',dtype=torch.float32)
        return 2*(x-self.lower)/self.span-1

    def evaluate(self,row,stage,*,force=False):
        if not row['valid']:
            self.costs[stage+'_rejections']+=1
            return None
        dh=row['design_hash']; cached=dh in self.cache and not force
        if not cached:
            event=dict(stage=stage,design_hash=dh,event='call_started',call_index=sum(v for k,v in self.costs.items() if k.endswith('_calls')))
            with (self.output/'Calls.jsonl').open('ab') as stream:
                stream.write(canonical(event)+b'\n'); stream.flush()
                import os
                os.fsync(stream.fileno())
            result=evaluate_geometry19_candidate(event['call_index'],stage,row['geometry19'],dh,self.target,self.acceptance,
                       self.context.physics_runtime.model_config,self.context.physics_runtime.constraint_profile,Environment(),
                       physics_runtime=self.context.physics_runtime)
            result['inside_target_domain']=bool(result['performance'] and self.context.target_domain.assess(result['performance']).inside)
            result['physics_acceptable']=result['physics_acceptable'] and result['inside_target_domain']
            result['development_diagnostic_hit']=result['development_diagnostic_hit'] and result['physics_acceptable']
            if not result['physics_acceptable']: result['hit_alignment']=None
            result=_json_safe(result)
            self.costs[stage+'_calls']+=1
            if not force: self.cache[dh]=deepcopy(result)
        else:
            result=deepcopy(self.cache[dh]); self.costs[stage+'_cache']+=1
        result['variant']=stage
        self.ledger.append(dict(stage=stage,cache_hit=cached,evaluation=result))
        if not force: self.physical_rows.append(result)
        return result

    def ranking(self,rows):
        g3.adapter._score(rows,self.public,self.dec,self.data['schema'],self.ids,self.scorer,
                          np.asarray([self.tolerances[n] for n in g3.adapter.OUTPUT_NAMES]))
        unique={}
        for row in rows:
            if row['valid']: unique.setdefault(row['design_hash'],row)
        return sorted(unique.values(),key=lambda r:(r['predicted_rho'],r['design_hash']))

    def generate(self):
        count=self.config['raw_candidates']
        source,_=g3.source_bank(self.config['generation_seed'],self.index,count)
        sources=torch.tensor(source,device='cuda:0')
        with torch.no_grad(): direct=integrate_cfm_sources(self.flow,sources,self.condition)
        guided,updated,history,metrics=_dflow_generate(self.flow,sources,direct,self.condition,self.target,self.protocol,
                            self.surrogate,self.stats,None,None,'hit_aligned')
        raw=self.decode(guided); self.direct_rows=self.decode(direct)
        self.ranked=self.ranking(raw)
        g3.seal_json(self.output/'B.json',dict(target=self.public,selected_expert=self.route,source=source.tolist(),
                    updated_source=updated.cpu().tolist(),direct_normalized=direct.cpu().tolist(),guided_normalized=guided.cpu().tolist(),
                    dflow_history=history,dflow_metrics=metrics,candidates=raw,ranking=[r['design_hash'] for r in self.ranked]))
        return self.ranked

    def stage_c(self):
        for row in self.ranked[:self.config['budgets']['C']]:
            result=self.evaluate(row,'C')
            if result is not None and result['development_diagnostic_hit']: return result
        return None

    def objective(self,x,origin):
        target=torch.tensor([self.public['performance'][n] for n in g3.adapter.OUTPUT_NAMES],device=x.device,dtype=x.dtype)
        return _surrogate_objective(x,origin,self.lower,self.lower+self.span,target,self.ref_models,
              self.protocol['dflow'],self.protocol['hit_alignment'],self.protocol['diagnostic_acceptance'],'hit_aligned')[0]

    def refine(self,seeds,stage='D'):
        if not seeds: return None
        current=torch.stack([self.normalized(s['geometry19']) for s in seeds[:8]])
        origin=current.clone(); prior=0
        baselines=[]
        for seed in seeds[:8]:
            cached=self.cache.get(seed['design_hash'])
            baselines.append(math.inf if cached is None or not cached['physics_acceptable'] else cached['hit_alignment']['rho'])
        for checkpoint in self.config['refinement']['checkpoints']:
            x=current.detach().clone().requires_grad_(True)
            optimizer=torch.optim.Adam([x],lr=self.protocol['dflow']['learning_rate'])
            for _ in range(checkpoint-prior):
                optimizer.zero_grad(); loss=self.objective(x,origin).mean()
                require(torch.isfinite(loss).item(),'Refinement loss nonfinite'); loss.backward()
                require(x.grad is not None and torch.isfinite(x.grad).all().item(),'Refinement gradient invalid')
                torch.nn.utils.clip_grad_norm_([x],1.); optimizer.step()
                with torch.no_grad(): x.copy_(project_local(x,self.flow.center,self.flow.radius))
            for i,row in enumerate(self.decode(x)):
                result=self.evaluate(row,stage)
                if result is not None and result['development_diagnostic_hit']: return result
                if result is not None and result['physics_acceptable'] and result['hit_alignment']['rho']<baselines[i]:
                    current[i]=x[i].detach(); baselines[i]=result['hit_alignment']['rho']
            prior=checkpoint
        return None

    def search(self,seeds,stage='E'):
        rng=np.random.default_rng(42+self.index)
        population=[]; evaluations=0
        stop_on_first_hit=bool(self.config.get('nsga2',{}).get('stop_on_first_hit',False))
        lower=np.full(16,-1.); upper=np.full(16,1.)
        warm=[self.normalized(r['geometry19']).cpu().numpy() for r in seeds if r['valid']]
        def evaluate(genes):
            nonlocal evaluations
            x=torch.tensor(np.asarray(genes),device='cuda:0',dtype=torch.float32)
            x=project_local(x,self.flow.center,self.flow.radius)
            def evaluate_one(item):
                vector,row=item
                nonlocal evaluations
                evaluations+=1
                value=self.evaluate(row,stage)
                return None if value is None else dict(genes=vector,evaluation=value)
            return evaluate_until_hit(zip(x.cpu().numpy(),self.decode(x)),evaluate_one,stop_on_first_hit)
        initial=warm[:16]
        while len(initial)<16:
            state=rng.normal(size=16); state*=rng.uniform(0,.15)/np.linalg.norm(state)
            initial.append(2*(self.flow.center.cpu().numpy()+self.flow.radius*state)-1)
        population,first_hit=evaluate(initial)
        if first_hit is not None and stop_on_first_hit: return first_hit
        for generation in range(16):
            hits=[r['evaluation'] for r in population if r['evaluation']['development_diagnostic_hit']]
            if hits: return min(hits,key=lambda r:r['hit_alignment']['rho'])
            if not population or generation==15 or evaluations+16>self.config['budgets']['E']: return None
            ranks,crowding=nsga._rank_and_crowding(population); children=[]
            while len(children)<16:
                left=population[nsga._tournament(ranks,crowding,rng)]['genes']
                right=population[nsga._tournament(ranks,crowding,rng)]['genes']
                for child in nsga._sbx(left,right,lower,upper,rng,rate=.9):
                    children.append(nsga._mutate(child,lower,upper,[],rng,rate=.1,scale=.01))
            new_population,first_hit=evaluate(children[:16])
            if first_hit is not None and stop_on_first_hit: return first_hit
            population=nsga._environmental_selection(population+new_population,16)
        return None

    def deliver(self,result):
        # JSON round-trip and force bypass of the within-task cache.
        g3.seal_json(self.output/'Delivery_Geometry.json',result['geometry19'])
        geometry=json.loads((self.output/'Delivery_Geometry.json').read_text())
        check=self.decode(self.normalized(geometry).reshape(1,16))[0]
        require(check['valid'],'Delivery support failed')
        # Recompute the exact stored doubles, not the float32 decoder round-trip.
        check.update(geometry19=geometry,design_hash=result['design_hash'])
        recalculated=self.evaluate(check,'F',force=True)
        passed=bool(recalculated['development_diagnostic_hit'] and performance_matches(result['performance'],recalculated['performance']))
        g3.seal_json(self.output/'F.json',dict(passed=passed,reference=result,recalculated=recalculated,cache_bypassed=True))
        return dict(passed=passed,design_hash=result['design_hash'])

    def derivative_audit(self):
        model=deepcopy(self.surrogate).double()
        z=self.flow.center.detach().double().requires_grad_(True)
        direction=torch.arange(1,17,device=z.device,dtype=z.dtype); direction/=direction.norm()
        analytic=torch.autograd.functional.jacobian(model,z)@direction
        h=1e-5
        numeric=(model(z+h*direction)-model(z-h*direction))/(2*h)
        error=float(((analytic-numeric).abs()/analytic.abs().clamp_min(1e-8)).max().item())
        require(error<.005,'Ridge derivative audit failed')
        full={k:v for k,v in zip(self.dec['integer_axes'],self.dec['experts'][self.route]['integer_values'])}
        for name,value in zip(self.dec['continuous_axes'],(self.lower+self.span*self.flow.center).cpu().tolist()): full[name]=value
        rows=[dict(valid=True,geometry19=full,support_root_id=self.dec['root_order'][self.route])]
        g3.adapter._score(rows,self.public,self.dec,self.data['schema'],self.ids,self.scorer,np.asarray([self.tolerances[n] for n in g3.adapter.OUTPUT_NAMES]))
        predicted=self.surrogate(self.flow.center).exp().detach().cpu().numpy()
        reference=np.asarray([rows[0]['predicted_performance'][n] for n in g3.adapter.OUTPUT_NAMES])
        relative=float(np.max(np.abs(predicted/reference-1)))
        require(relative<2e-5,'Differentiable Ridge changed predictions')
        return dict(finite_difference_relative_error=error,numpy_prediction_relative_error=relative)


def run(root):
    root=Path(root).resolve(); config=yaml.safe_load((root/CONFIG).read_text()); data=g3.preflight(root)
    device=g3.adapter._cuda_manifest()
    models,_=g3.adapter.load_seed_models(root,data['inference'],42,data['decoders'],'cuda:0')
    ids,scorer=g3._load_model(data['paths']['s2_oracle_local_ridge'],'stable_38_oracle_root_local_ridge')
    output=root/config['output']; output.mkdir(parents=True,exist_ok=False)
    code=[CONFIG,'src/ai/geometry19_local_full_pipeline.py','scripts/run_geometry19_local_full_pipeline.py','tests/test_geometry19_local_full_pipeline.py',
          'src/ai/geometry19_candidates.py','src/ai/geometry19_refinement.py','src/ai/geometry19_nsga2.py','src/ai/geometry19_delivery.py','src/ai/geometry19_physics.py','src/ai/geometry19_cfm.py']
    lock=dict(protocol=config,code={p:file_hash(root/p) for p in code},inputs=data['inputs'],runtime=data['context'].contract_mapping(),device=device)
    g3.seal_json(output/'Protocol_Lock.json',lock)
    results=[]
    for index in config['targets']:
        started=time.monotonic(); task=LocalTask(data,config,models,ids,scorer,index,output/f'target{index:03d}')
        gradient=task.derivative_audit(); task.generate()
        result=execute_stages(task.stage_c,lambda:task.refine(task.ranked[:8]),lambda:task.search(task.ranked),task.deliver)
        probes={}
        if index==config['component_probes']['target']:
            direct=[r for r in task.direct_rows if r['valid']]
            require(bool(direct),'No direct probe seed')
            d=task.refine(direct[:1],'probe_D'); e=task.search(direct,'probe_E')
            probes=dict(D_executed=task.costs['probe_D_calls']>0,E_executed=task.costs['probe_E_calls']>0,
                        D_hit=d is not None,E_hit=e is not None,counts_as_pipeline_rescue=False)
        g3.seal_json(task.output/'Physics_Ledger.json',task.ledger)
        row=dict(target_index=index,target_class=data['targets'][index]['target_class'],flow=result,component_probes=probes,
                 gradient_audit=gradient,costs=dict(task.costs),elapsed_seconds=time.monotonic()-started)
        g3.seal_json(task.output/'Summary.json',row); results.append(row)
        print(json.dumps(row),flush=True)
    summary=dict(status='N3_LOCAL_FULL_FLOW_SMOKE_COMPLETED',tasks=results,
                 all_delivery_pass=all(r['flow']['delivery'] and r['flow']['delivery']['passed'] for r in results),
                 production_B_F_entry_integrated=False,full_99_target_comparison_executed=False,
                 reuse='V2.1_DFlow_gradient_objective_NSGA_operators_physics_evaluator_delivery_comparison')
    g3.seal_json(output/'Summary.json',summary)
    g3.seal_json(output/'Complete_Manifest.json',dict(files={str(p.relative_to(output)):file_hash(p) for p in output.rglob('*') if p.is_file()}))
    return summary
