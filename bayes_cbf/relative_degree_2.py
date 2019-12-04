import numpy as np
import torch
from contextlib import contextmanager
from collections import namedtuple

from bayes_cbf.control_affine_model import GaussianProcess

def torchify_method(method):
    for k, v in vars(method.__self__).items():
        if isinstance(v, (float, np.ndarray)):
            setattr(method.__self__, k, torch.from_numpy(np.asarray(v)))

def untorchify_method(method):
    for k, v in vars(method.__self__).items():
        if isinstance(v, (torch.Tensor,)):
            setattr(method.__self__, k, np.asarray(v))

def torchified_method(method):
    if not hasattr(method, "__self__"):
        # do nothing
        return method

    try:
        yield torchify_method(method)
    finally:
        untorchify_method(method)

def get_torch_grad_from_numpy_method(method, x):
    xtorch = torch.from_numpy(x).requires_grad_(True)
    with torchify_method(method) as tmethod:
        tmethod(xtorch).backward()
        return xtorch.grad

def affine_gp(affine, f):
    """
    return aᵀf, aᵀk_fa, k_fa
    """
    mean_a = affine.T @ f.mean
    k_a = affine.T @ f.k @ affine
    covar_af = f.k @ affine.T
    return GaussianProcess(mean=mean_a, k=k_a, covar=covar_af)


def t_jac(f_x, x):
    return torch.cat(
        [torch.autograd.grad(f_x[i], x, retain_graph=True)[0].unsqueeze(0)
         for i in range(f_x.shape[0])], dim=0)


def grad_gp(f, x):
    grad_k = torch.autograd.grad(f.k, x, create_graph=True)[0]
    Hxx_k_v_prod = lambda v: torch.autograd.grad(grad_k, x, grad_outputs=v,
                                                 retain_graph=True)[0]
    y = x.clone().detach().requires_grad_(False)
    covar_f = f.covar
    J_covar_f = t_jac(covar_f, x)
    return GaussianProcess(mean=torch.autograd.grad(f.mean, x, retain_graph=True)[0],
                           k=Hxx_k_v_prod,
                           covar=J_covar_f)


def quadratic_form_of_gps(g, f):
    """
    s = fᵀ g = fᵀ [0, 0.5]
                  [0.5, 0] g


    Given mean and variance of generic random variables
    https://link.springer.com/content/pdf/10.1007%2F978-1-4612-4242-0.pdf

    On products of Gaussian random variables
    https://arxiv.org/pdf/1711.10516.pdf
    https://projecteuclid.org/download/pdf_1/euclid.ecp/1465320999

    More referennces:

    1. Concentration inqualities for polynomials in α-sub-exponential random variables
    By Gotze, Sambale and Sinulis
    https://arxiv.org/pdf/1903.05964.pdf

    2. Norm of sub-exponential random variables

    3. A note on the Hanson-Wright inequality for random vectors with depdendencies.
       By Radoslaw Adamczak

    From Hansen-Wright inequality for dependent random variables:
    The result of the quadratic form on Gaussian RV is sub-exponential upper bound.

    From extensions of Hansen-Wright inequalities for sub-exponential variables
    ( Tail decay faster than (α exp(-t) ) )
    The result of the quadratic form on Sub-exponential RV is a tail that
    decays (α exp(-t) / 2)
    """
    covar_xf = g.covar
    mean_quad = g.mean @ f.mean + covar_xf.trace()
    k_quad = (2 * covar_xf.trace()
              + 2 * g.mean.T @ covar_xf @ f.mean
              + g.mean.T @ f.k @ f.mean
              + f.mean.T @ g.k(f.mean))
    covar_quad_f = covar_xf @ f.mean + f.k @ g.mean
    return GaussianProcess(mean=mean_quad, k=k_quad, covar=covar_quad_f)

def add_gp(f, g):
    """
    s = f + g
    """
    return GaussianProcess(mean=f.mean + g.mean, k=f.k + g.k + 2 * f.covar, covar=f.k)


def lie1_gp(h, fu, x):
    """
    L_f h(x) = ∇ h(x)ᵀ f(x; u)
    """
    grad_h = torch.autograd.grad(h, x, retain_graph=True, create_graph=True)[0]
    return affine_gp(grad_h, fu)


def lie2_gp(h_func, fu_func, x, u):
    """
    L_f² h(x) = ∇[L_f h(x)]ᵀ f(x; u)
    """
    x = x.requires_grad_(True)
    h = h_func(x)
    fu = fu_func(x, u)
    L1h = lie1_gp(h, fu, x)
    grad_L1h = grad_gp(L1h, x)
    return quadratic_form_of_gps(grad_L1h, fu)


def cbc2_gp(h_func, fu_func, x, u, K_α=[1,1]):
    """
    L_f² h(x) + K_α [     h(x) ] ≥ 0
                    [ L_f h(x) ]
    """
    x.requires_grad_(True)
    u.requires_grad_(True)
    L1h = lie1_gp(h_func(x), fu_func(x, u), x)
    L2h = lie2_gp(h_func, fu_func, x, u)
    h = h_func(x)
    return add_gp(L2h, GaussianProcess(K_α[1] * L1h.mean + K_α[0] * h,
                                       K_α[1] * K_α[1] * L1h.k,
                                       K_α[1] * L1h.covar))

def get_affine_terms(affine_func, x):
    x = x.requires_grad_(True)
    f_x = func(x)
    linear = torch.autograd.grad(f_x, x, create_graph=True)[0]
    with torch.no_grad():
        const = f_ - linear @ x
    return linear, const


def get_quadratic_terms(func, x):
    x = x.requires_grad_(True)
    f_x = func(x)
    linear_more = torch.autograd.grad(f_x, x, create_graph=True)[0]
    quad = t_jac(linear_more, x) / 2
    with torch.no_grad():
        linear = linear_more - 2 * quad @ x
        const = f_x - x.T @ quad @ x - linear @ x
    return quad, linear, const


def cbf2_quadratic_constraint(h_func, fu_func, x, u, δ):
    """
    cbc2.mean(x) ≥ √(1-δ)/δ cbc2.k(x,x)

    """
    mean_func = lambda up : cbc2_gp(h_func, fu_func, x, up).mean
    mean_A, mean_b = get_affine_terms(mean_func, u)
    k_func = lambda up : cbc2_gp(h_func, fu_func, x, up).k
    k_Q, k_p, k_r = get_quadratic_terms(k_func, u)
    ratio = (1-δ) / δ
    return ratio * k_Q, ratio * k_p - mean_A, ratio * k_r - mean_b




