"""V3 target/center conditional flow matching in a frozen 19-continuous-dimensional chart."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from torch import nn
import yaml
from src.ai.v3_extended_gate_l6f import config_at, source_data, sha
from src.physics.geometry22_data import Geometry22Normalizer
from src.physics.v3_development_runtime import build_v3_development_physics_runtime

CONFIG = "config/experiments/geometry22_v3_local_cfm_v1.yaml"
CODE = "src/ai/v3_local_cfm.py"
NAMES = ('thrust_N', 'tsfc_kg_N_s', 'mass_kg')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


class LocalCFM(nn.Module):
    def __init__(self, centers=55, width=128, embedding=16):
        super().__init__()
        self.embedding = nn.Embedding(centers, embedding)
        self.velocity_net = nn.Sequential(nn.Linear(19+1+3+embedding, width), nn.SiLU(),
                                          nn.Linear(width, width), nn.SiLU(), nn.Linear(width,19))

    def forward(self, x, t, condition, center):
        return self.velocity_net(torch.cat((x,t.reshape(-1,1),condition,self.embedding(center)),dim=1))


@torch.no_grad()
def integrate(model, noise, condition, center, steps=32):
    require(steps > 0, 'Invalid ODE steps')
    x = noise.clone()
    for index in range(steps):
        t = torch.full((len(x),), index/steps, device=x.device)
        first = model(x,t,condition,center)
        middle = model(x+first/(2*steps),t+0.5/steps,condition,center)
        x = x+middle/steps
    return x


def device():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '6' and torch.cuda.is_available()
            and torch.cuda.device_count() == 1, 'Local CFM requires physical GPU 6; no CPU fallback')
    torch.use_deterministic_algorithms(True)
    return torch.device('cuda:0')


def cfg_at(root):
    cfg = yaml.safe_load((root/CONFIG).read_text())
    require(cfg['version']=='v3_local_cfm_v1' and cfg['center_count']==55 and cfg['radius']==.08
            and cfg['training_scope']=='development_internal_fit_only' and cfg['independent_validation'] is False
            and cfg['formal_use_allowed'] is False and cfg['deployment_allowed'] is False, 'CFM scope changed')
    return cfg


def dataset(root,cfg):
    targets,designs,_ = source_data(root,config_at(root))
    assets=root/cfg['l6f_root']
    support=json.loads((assets/'center_support.json').read_text())['centers']
    norm=json.loads((assets/'normalization.json').read_text())
    continuous=[i for i,v in enumerate(Geometry22Normalizer().fields) if v['dtype']!='int64']
    require(len(continuous)==19,'Expected 19 continuous and 3 integer variables')
    by_id={r['design_id']:r['normalized_design'] for r in designs}
    classes=[s['center_id'] for s in support]
    centers={s['center_id']:s for s in support}
    x,y,c,fit=[],[],[],[]
    for row in targets:
        source=centers[row['source_center_id']]
        design=np.asarray(by_id[row['target_id']],dtype=np.float64)
        anchor=np.asarray(source['center_normalized_design'],dtype=np.float64)
        integer=[i for i in range(22) if i not in continuous]
        require(np.allclose(design[integer],anchor[integer],rtol=0,atol=1e-12),'CFM integer chart changed')
        residual=(design[continuous]-anchor[continuous])/cfg['radius']
        require(np.isfinite(residual).all() and np.linalg.norm(residual)<=1+1e-8,'CFM training outside chart')
        x.append(residual)
        y.append((np.log([row['target_performance'][n] for n in NAMES])-norm['log_mean'])/norm['log_std'])
        c.append(classes.index(row['source_center_id']))
        fit.append(row['internal_split']=='fit')
    arrays=(np.asarray(x,dtype=np.float32),np.asarray(y,dtype=np.float32),np.asarray(c,dtype=np.int64),np.asarray(fit))
    require(int(arrays[-1].sum())==cfg['fit_rows'] and int((~arrays[-1]).sum())==cfg['tuning_rows'],'CFM split changed')
    return arrays,classes,continuous,targets


def identity(root,cfg):
    # Bind the actual data readers, split, gate assets and schema/physics contract.
    source_cfg=config_at(root)
    sources={CONFIG:sha(root/CONFIG),CODE:sha(root/CODE),
             'src/ai/v3_extended_gate_l6f.py':sha(root/'src/ai/v3_extended_gate_l6f.py'),
             'src/ai/v3_target_gate_l6b.py':sha(root/'src/ai/v3_target_gate_l6b.py'),
             'src/physics/geometry22_data.py':sha(root/'src/physics/geometry22_data.py')}
    for filename in ('fit_manifest.json','normalization.json','center_support.json','target_registry.jsonl'):
        name=cfg['l6f_root']+'/'+filename
        sources[name]=sha(root/name)
    return {'config':cfg,'source_sha256':sources,'source_data_contract':source_cfg,
            'runtime_contract':build_v3_development_physics_runtime(root).contract_mapping(),
            'physical_gpu':6,'validation_holdout_used_for_training':False}


def freeze(root):
    cfg=cfg_at(root)
    arrays,classes,continuous,targets=dataset(root,cfg)
    path=root/cfg['assets_root']
    require(not path.exists(),'Refusing to overwrite CFM assets')
    path.mkdir(parents=True)
    write(path/'protocol_lock.json',identity(root,cfg))
    write(path/'dataset_manifest.json',{'fit_rows':cfg['fit_rows'],'tuning_rows':cfg['tuning_rows'],
          'class_ids':classes,'continuous_indices':continuous,'design_dimension':22,
          'target_ids_sha256':hashlib.sha256(json.dumps([(r['target_id'],r['internal_split']) for r in targets]).encode()).hexdigest(),
          'target_normalization':'frozen_l6f_fit_only','integer_policy':'fixed_from_routed_center',
          'training_scope':'software_development','formal_use_allowed':False,'deployment_allowed':False})
    return {'status':'frozen_before_training','fit_rows':cfg['fit_rows'],'tuning_rows':cfg['tuning_rows']}


def fit(root):
    cfg=cfg_at(root);path=root/cfg['assets_root']
    require(json.loads((path/'protocol_lock.json').read_text())==identity(root,cfg),'CFM frozen identity changed')
    require(not (path/'checkpoint.pt').exists(),'Refusing to retrain/overwrite CFM checkpoint')
    d=device();torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    (x,y,c,mask),classes,continuous,_=dataset(root,cfg)
    x=torch.tensor(x,device=d);y=torch.tensor(y,device=d);c=torch.tensor(c,device=d)
    fit_ids=torch.tensor(np.flatnonzero(mask),device=d);tune_ids=torch.tensor(np.flatnonzero(~mask),device=d)
    model=LocalCFM(cfg['center_count'],cfg['hidden_width'],cfg['embedding_dimension']).to(d)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
    counts=torch.bincount(c[fit_ids],minlength=55).float()
    weights=counts.mean()/counts
    fixed_noise=torch.randn((len(tune_ids),19),device=d)*cfg['noise_std']
    fixed_t=torch.linspace(.05,.95,len(tune_ids),device=d)
    history=[];best=float('inf');best_state=None;best_epoch=None
    for epoch in range(cfg['epochs']):
        model.train();loss_sum=0
        order=fit_ids[torch.randperm(len(fit_ids),device=d)]
        for batch in order.split(cfg['batch_size']):
            noise=torch.randn_like(x[batch])*cfg['noise_std'];t=torch.rand(len(batch),device=d)
            state=(1-t[:,None])*noise+t[:,None]*x[batch]
            errors=(model(state,t,y[batch],c[batch])-(x[batch]-noise)).square().mean(1)
            loss=(errors*weights[c[batch]]).mean()
            require(torch.isfinite(loss).item(),'Nonfinite CFM loss')
            optimizer.zero_grad();loss.backward();optimizer.step();loss_sum+=float(loss)*len(batch)
        model.eval()
        with torch.no_grad():
            state=(1-fixed_t[:,None])*fixed_noise+fixed_t[:,None]*x[tune_ids]
            errors=(model(state,fixed_t,y[tune_ids],c[tune_ids])-(x[tune_ids]-fixed_noise)).square().mean(1)
            per_center=[float(errors[c[tune_ids]==i].mean()) for i in range(55)]
            score=float(np.mean(per_center))
        history.append({'epoch':epoch+1,'fit_velocity_mse':loss_sum/len(fit_ids),'tuning_macro_velocity_mse':score})
        if score<best:
            best=score;best_epoch=epoch+1;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if (epoch+1)%10==0:print(f'CFM epoch {epoch+1}/{cfg["epochs"]}: tuning={score:.6g}',flush=True)
    checkpoint={'state_dict':best_state,'class_ids':classes,'continuous_indices':continuous,'config':cfg,
                'design_schema_hash':identity(root,cfg)['runtime_contract']['design_schema_hash'],
                'physics_bundle_hash':identity(root,cfg)['runtime_contract']['physics_bundle_hash']}
    torch.save(checkpoint,path/'checkpoint.pt')
    write(path/'fit_report.json',{'status':'local_cfm_development_trained','best_epoch':best_epoch,'best_tuning_macro_velocity_mse':best,
          'history':history,'physical_gpu':6,'device_name':torch.cuda.get_device_name(0),
          'independent_generalization_claim':False,'formal_use_allowed':False,'deployment_allowed':False})
    files={p.name:sha(p) for p in path.iterdir() if p.is_file()}
    write(path/'complete_manifest.json',{'files':files})
    return verify(root)


def verify(root):
    cfg=cfg_at(root);path=root/cfg['assets_root'];seal=json.loads((path/'complete_manifest.json').read_text())
    require(set(seal['files'])=={p.name for p in path.iterdir() if p.is_file() and p.name!='complete_manifest.json'},'CFM asset inventory changed')
    require(all(sha(path/name)==digest for name,digest in seal['files'].items()),'CFM asset changed')
    require(json.loads((path/'protocol_lock.json').read_text())==identity(root,cfg),'CFM training identity changed')
    return json.loads((path/'fit_report.json').read_text()) | {'integrity_pass':True,'checkpoint_sha256':sha(path/'checkpoint.pt')}


def load(root, device):
    cfg=cfg_at(root);verified=verify(root)
    payload=torch.load(root/cfg['assets_root']/'checkpoint.pt',map_location='cpu',weights_only=True)
    require(payload['config']==cfg,'CFM checkpoint config changed')
    model=LocalCFM(cfg['center_count'],cfg['hidden_width'],cfg['embedding_dimension']).to(device)
    model.load_state_dict(payload['state_dict']);model.eval()
    return model,payload,verified


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','fit','verify'])
    args=parser.parse_args();root=Path(__file__).resolve().parents[2]
    print(json.dumps(globals()[args.command](root)))
