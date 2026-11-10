import pytest
from src.ai.geometry19_local_full_pipeline import execute_stages, project_local
import torch


@pytest.mark.parametrize('first',['C','D','E',None])
def test_conditional_stages_and_independent_delivery(first):
    seen=[]
    def stage(name):
        def call():
            seen.append(name)
            return {'development_diagnostic_hit':name==first}
        return call
    def deliver(result):
        assert result['development_diagnostic_hit']; seen.append('F'); return {'passed':True}
    result=execute_stages(stage('C'),stage('D'),stage('E'),deliver)
    expected=['C','D','E'] if first is None else ['C','D','E'][:['C','D','E'].index(first)+1]+['F']
    assert seen==expected and result['visited']==expected and result['first_hit_stage']==first


def test_delivery_failure_is_not_success():
    def hit(): return {'development_diagnostic_hit':True}
    result=execute_stages(hit,lambda:pytest.fail('D ran after C hit'),lambda:pytest.fail('E ran after C hit'),lambda r:{'passed':False})
    assert result['delivery']['passed'] is False


def test_local_projection_respects_both_box_and_ball():
    center=torch.tensor([0.,1.,.5]); radius=.1
    x=torch.tensor([[-9.,9.,5.]])
    z=(project_local(x,center,radius)+1)/2
    assert (z>=0).all() and (z<=1).all()
    assert torch.linalg.vector_norm(z-center)<=radius*.20+1e-7
