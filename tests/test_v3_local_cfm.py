from pathlib import Path
import json
import numpy as np
import pytest
import torch
from src.ai.v3_local_cfm import LocalCFM, integrate, verify
from src.ai.geometry22_pipeline_v3_cfm import decode_candidate, source
from src.physics.geometry22_data import Geometry22Normalizer

ROOT=Path(__file__).resolve().parents[1]

def test_velocity_depends_on_target_and_center_and_has_trainable_gradients():
    torch.manual_seed(1)
    model=LocalCFM()
    x=torch.zeros(2,19);t=torch.ones(2)*.5
    a=model(x,t,torch.zeros(2,3),torch.zeros(2,dtype=torch.long))
    b=model(x,t,torch.ones(2,3),torch.zeros(2,dtype=torch.long))
    c=model(x,t,torch.zeros(2,3),torch.ones(2,dtype=torch.long))
    assert not torch.equal(a,b) and not torch.equal(a,c)
    a.square().mean().backward()
    assert model.embedding.weight.grad.abs().sum()>0


def test_midpoint_constant_velocity_integrates_exactly_without_mutating_noise():
    class Constant:
        def __call__(self,x,t,c,k):return torch.ones_like(x)*.25
    noise=torch.zeros(4,19)
    result=integrate(Constant(),noise,torch.zeros(4,3),torch.zeros(4,dtype=torch.long),32)
    assert torch.allclose(result,torch.full_like(noise,.25)) and torch.count_nonzero(noise)==0


def test_cfm_decode_preserves_22d_identity_integer_state_and_rejects_outside_ball():
    cfg,_,local,support,_,_=source(ROOT)
    center=support[0];norm=Geometry22Normalizer();continuous=local['input_continuous_indices_zero_based']
    public={'target_id':'test'}
    row=decode_candidate(public,0,center,np.zeros(19),norm,continuous,cfg)
    assert row['status']=='geometric_candidate' and len(row['design'])==22
    assert all(row['design'][n]==v for n,v in center['integer_state'].items())
    outside=decode_candidate(public,1,center,np.ones(19),norm,continuous,cfg)
    assert outside['status']=='outside_local_support' and 'design' not in outside
    invalid=decode_candidate(public,2,center,np.full(19,np.nan),norm,continuous,cfg)
    assert invalid['status']=='outside_local_support' and invalid['normalized_radius'] is None


def test_trained_assets_are_sealed_and_only_development_internal_split():
    report=verify(ROOT)
    assert report['integrity_pass'] and report['physical_gpu']==6
    data=json.loads((ROOT/'data/validation/v3_local_cfm_v1/dataset_manifest.json').read_text())
    assert data['fit_rows']==21665 and data['tuning_rows']==5513 and len(data['class_ids'])==55
    lock=json.loads((ROOT/'data/validation/v3_local_cfm_v1/protocol_lock.json').read_text())
    assert lock['validation_holdout_used_for_training'] is False


def test_checkpoint_tamper_rejected_before_loading(tmp_path, monkeypatch):
    import shutil
    from src.ai import v3_local_cfm as module
    cfg=module.cfg_at(ROOT)
    shutil.copytree(ROOT/cfg['assets_root'],tmp_path/'assets')
    changed=dict(cfg,assets_root='assets')
    lock=json.loads((tmp_path/'assets/protocol_lock.json').read_text())
    monkeypatch.setattr(module,'cfg_at',lambda _:changed)
    monkeypatch.setattr(module,'identity',lambda *_:lock)
    assert module.verify(tmp_path)['integrity_pass']
    with (tmp_path/'assets/checkpoint.pt').open('ab') as stream:stream.write(b'tampered')
    with pytest.raises(ValueError,match='asset changed'):
        module.verify(tmp_path)


def test_wrong_physical_gpu_never_falls_back_to_cpu(monkeypatch):
    from src.ai.v3_local_cfm import device
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    with pytest.raises(ValueError,match='physical GPU 6'):
        device()
