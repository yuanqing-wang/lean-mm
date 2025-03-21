import torch

def action_matching_loss(
    unbiasing_force: torch.nn.Module,
    samples: torch.Tensor,
    times: torch.Tensor,  
    weights: torch.Tensor,
    T: float=1.0,
    epsilon: float=1.0,
    time_scale: float=1.0,
) -> torch.Tensor:

    # set requires_grad
    samples.requires_grad = True
    times.requires_grad = True
        
    # set the scale
    epsilon = epsilon.unsqueeze(-1).unsqueeze(-1)
    unitless_unbiasing_force = lambda x, t: unbiasing_force(x, t, scale=epsilon)
        
    # take gradient
    df_dt, df_dx = torch.autograd.grad(
        torch.vmap(torch.vmap(unitless_unbiasing_force))(samples, times).sum(),
        [times, samples],
        create_graph=True,
    )
    
    # sum the force
    df_dx = (df_dx**2).sum(-1).sum(-1)
    
    # remove the unit
    df_dt = df_dt
    df_dx = df_dx * time_scale
    
    # compute the loss
    loss = 0.5 * df_dx + df_dt
    
    # normalize weights
    weights = weights.softmax(0)
    
    # compute the weighted loss
    loss = weights * loss
    
    # compute initial and final energy
    f0 = torch.vmap(unitless_unbiasing_force)(samples[:, 0], torch.zeros(len(samples)))
    f1 = torch.vmap(unitless_unbiasing_force)(samples[:, -1], torch.ones(len(samples))*T)
    f1 = weights[:, -1] * f1
        
    # combine the loss
    loss = loss.sum(-1) + f0 - f1
    
    # combine the loss
    loss = loss.mean()
    return loss
    

    
    