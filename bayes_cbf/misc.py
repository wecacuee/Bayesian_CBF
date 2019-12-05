"""
Home for functions/classes that haven't find a home of their own
"""
import torch

def t_vstack(xs):
    torch.cat(xs, dim=-2)

def to_numpy(x):
    return x.detach().cpu().double().numpy()

def t_jac(f_x, x):
    if f_x.ndim:
        return torch.cat(
            [torch.autograd.grad(f_x[i], x, retain_graph=True)[0].unsqueeze(0)
            for i in range(f_x.shape[0])], dim=0)
    else:
        return torch.autograd.grad(f_x, x, retain_graph=True)[0]


