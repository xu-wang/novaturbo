"""Differentiable CFM source guidance and projected local-coordinate refinement."""
import torch


def flow(model,x,condition,center,steps):
    # Unlike inference integrate(), retain the graph all the way to the source.
    for index in range(steps):
        t=torch.full((len(x),),index/steps,device=x.device)
        v=model(x,t,condition,center)
        x=x+model(x+v/(2*steps),t+.5/steps,condition,center)/steps
    return x


def objective(residual, coefficients, intercept, target):
    log_prediction=residual@coefficients+intercept
    return (log_prediction-torch.log(target)).square().mean(1)


def project(x, anchor, continuous, radius=.08):
    lower=(-1-anchor[continuous])/radius
    upper=(1-anchor[continuous])/radius
    clipped=torch.maximum(torch.minimum(x,upper),lower)
    return clipped/torch.linalg.vector_norm(clipped,dim=1,keepdim=True).clamp_min(1.)


def guide(model,noise,condition,center,coefficients,intercept,target,steps=8,lr=.03,ode_steps=32):
    for parameter in model.parameters():parameter.requires_grad_(False)
    initial=noise.detach().clone()
    x=initial.clone().requires_grad_(True)
    optimizer=torch.optim.Adam([x],lr=lr)
    history=[]
    with torch.no_grad():
        best=initial.clone()
        baseline=objective(flow(model,initial,condition,center,ode_steps),coefficients,intercept,target)
        best_loss=baseline.clone()
    for step in range(steps):
        optimizer.zero_grad()
        generated=flow(model,x,condition,center,ode_steps)
        losses=objective(generated,coefficients,intercept,target)
        loss=(losses+.001*(x-initial).square().mean(1)).mean()
        if not torch.isfinite(loss):raise ValueError('Nonfinite D-Flow objective')
        loss.backward()
        if x.grad is None or not torch.isfinite(x.grad).all():raise ValueError('Invalid D-Flow source gradient')
        torch.nn.utils.clip_grad_norm_([x],1.)
        optimizer.step()
        with torch.no_grad():
            current=objective(flow(model,x,condition,center,ode_steps),coefficients,intercept,target)
            better=torch.isfinite(current)&(current<best_loss)
            best[better]=x[better];best_loss[better]=current[better]
            history.append({'step':step+1,'mean_best_log_mse':float(best_loss.mean())})
    return flow(model,best,condition,center,ode_steps).detach(), {'source_updates':steps,'initial_log_mse':float(baseline.mean()),'final_log_mse':float(best_loss.mean()),'history':history}


def refine(residual,anchor,continuous,coefficients,intercept,target,steps=24,lr=.03):
    initial=residual.detach().clone()
    x=initial.clone().requires_grad_(True)
    optimizer=torch.optim.Adam([x],lr=lr)
    with torch.no_grad():best=initial.clone();best_loss=objective(best,coefficients,intercept,target)
    history=[]
    for step in range(steps):
        optimizer.zero_grad();loss=objective(x,coefficients,intercept,target).mean()
        if not torch.isfinite(loss):raise ValueError('Nonfinite refinement objective')
        loss.backward()
        if x.grad is None or not torch.isfinite(x.grad).all():raise ValueError('Invalid continuous gradient')
        torch.nn.utils.clip_grad_norm_([x],1.);optimizer.step()
        with torch.no_grad():
            x.copy_(project(x,anchor,continuous))
            values=objective(x,coefficients,intercept,target)
            better=values<best_loss;best[better]=x[better];best_loss[better]=values[better]
            history.append({'step':step+1,'mean_best_log_mse':float(best_loss.mean())})
    return best.detach(),history
