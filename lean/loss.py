import torch

def action_matching_loss(
    unbiasing_force: torch.nn.Module,
    samples: torch.Tensor,
    times: torch.Tensor,  
    weights: torch.Tensor,
    T: float=1.0,
) -> torch.Tensor:
    
    # take gradient
    df_dt, df_dx = torch.autograd.grad(
        unbiasing_force(samples, times).sum(),
        [times, samples],
        create_graph=True,
    )
    
    # summarize the gradient
    df_dx = (df_dx**2).sum(-1)
    
    # compute the loss
    loss = 0.5 * df_dx + df_dt
    
    # normalize weights
    weights = weights.softmax(0)
    
    # compute the weighted loss
    loss = weights * loss
    
    # compute initial and final energy
    f0 = unbiasing_force(samples, torch.zeros_like(times))
    f1 = unbiasing_force(samples, torch.ones_like(times)*T)
    f1 = weights * f1
    
    # combine the loss
    loss = loss + f0 - f1
    
    # combine the loss
    loss = loss.mean()
    return loss
    

    
    