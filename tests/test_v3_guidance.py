import torch
import pytest
from src.ai.v3_guidance import guide, refine, objective, project


def tensors():
    coefficients=torch.zeros(19,3);coefficients[:3]=torch.eye(3)
    return coefficients,torch.zeros(3)


def test_dflow_differentiates_through_ode_and_improves_source_objective():
    class Stationary(torch.nn.Module):
        def forward(self,x,t,c,k):return x*0
    coef,intercept=tensors();noise=torch.zeros(4,19)
    target=torch.exp(torch.ones(3)*.15)
    output,history=guide(Stationary(),noise,torch.zeros(4,3),torch.zeros(4,dtype=torch.long),coef,intercept,target,8,.03,4)
    assert history['final_log_mse']<history['initial_log_mse']
    assert not torch.equal(output,noise) and torch.count_nonzero(noise)==0
    assert all(b['mean_best_log_mse']<=a['mean_best_log_mse'] for a,b in zip(history['history'],history['history'][1:]))


def test_refinement_updates_continuous_design_inside_ball_and_box():
    coef,intercept=tensors();anchor=torch.zeros(22);continuous=list(range(19));x=torch.ones(2,19)*.1
    target=torch.ones(3)
    output,history=refine(x,anchor,continuous,coef,intercept,target,24,.03)
    assert objective(output,coef,intercept,target).mean()<objective(x,coef,intercept,target).mean()
    assert torch.all(torch.linalg.vector_norm(output,dim=1)<=1.)
    projected=project(torch.ones(2,19)*10,anchor,continuous)
    assert torch.all(torch.linalg.vector_norm(projected,dim=1)<=1.+1e-6)


def test_nonfinite_gradient_objective_stops_explicitly():
    coef,intercept=tensors()
    with pytest.raises(ValueError,match='Nonfinite'):
        refine(torch.zeros(2,19),torch.zeros(22),list(range(19)),coef,intercept,torch.full((3,),float('nan')))


def test_linear_flow_preserves_source_gradient_through_all_ode_steps():
    from src.ai.v3_guidance import flow
    class Contracting(torch.nn.Module):
        def forward(self,x,t,c,k):return -x
    x=torch.ones(2,19,requires_grad=True)
    output=flow(Contracting(),x,torch.zeros(2,3),torch.zeros(2,dtype=torch.long),32)
    output.sum().backward()
    assert torch.allclose(x.grad,torch.full_like(x,.367879),atol=1e-4)
