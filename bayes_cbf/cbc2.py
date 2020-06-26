import torch
from .gp_algebra import *

def cbc2_quadratic_terms(h_func, grad_h_func, control_affine_model, x, u, k_α):
    """
    cbc2.mean(x) ≥ √(1-δ)/δ cbc2.k(x,x')
    """
    # TODO: Too complicated and opaque. Try to find a way to simplify
    mean = lambda up: cbc2_gp(h_func, grad_h_func, control_affine_model, up, k_α).mean(x)
    k_func = lambda up: cbc2_gp(h_func, grad_h_func, control_affine_model, up, k_α).knl(x, x)

    #assert mean(u) > 0, 'cbf2 should be at least satisfied in expectation'
    mean_A, mean_b = get_affine_terms(mean, u)
    assert not torch.isnan(mean_A).any()
    assert not torch.isnan(mean_b).any()
    k_Q, k_p, k_r = get_quadratic_terms(k_func, u)
    assert not torch.isnan(k_Q).any()
    assert not torch.isnan(k_p).any()
    assert not torch.isnan(k_r).any()
    return (mean_A, mean_b), (k_Q, k_p, k_r), mean(u), k_func(u)


def cbc2_gp(h, grad_h, learned_model, utest, k_α):
    f_gp = learned_model.f_func_gp()
    fu_gp = learned_model.fu_func_gp(utest)
    h_gp = DeterministicGP(h, shape=(1,), name="h(x)")
    grad_h_gp = DeterministicGP(grad_h, shape=(learned_model.state_size,), name="∇ h(x)")
    L1h = grad_h_gp.t() @ f_gp
    L2h = GradientGP(L1h, x_shape=(learned_model.state_size,)).t() @ fu_gp
    return L2h + h_gp * k_α[0] + L1h * k_α[1]


def get_affine_terms(func, x):
    with variable_required_grad(x):
        f_x = func(x)
        linear = torch.autograd.grad(f_x, x, create_graph=True)[0]
    with torch.no_grad():
        const = f_x - linear @ x
    return linear, const


def get_quadratic_terms(func, x):
    with variable_required_grad(x):
        f_x = func(x)
        linear_more = torch.autograd.grad(f_x, x, create_graph=True)[0]
        quad = t_jac(linear_more, x) / 2
    with torch.no_grad():
        linear = linear_more - 2 * quad @ x
        const = f_x - x.T @ quad @ x - linear @ x
    return quad, linear, const

