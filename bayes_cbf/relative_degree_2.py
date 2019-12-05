import numpy as np
import torch
from contextlib import contextmanager
from collections import namedtuple

from bayes_cbf.control_affine_model import GaussianProcess
from bayes_cbf.misc import t_jac

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

class FunctionGroup:
    def __getattribute__(self, name):
        val = object.__getattr__(self, name)
        if not isinstance(val, classmethod):
            return staticmethod(val)

class GPFunctionGroup:
    @classmethod
    def gp(cls, *args):
        return (GaussianProcess(mean=cls.mean(*args), k=cls.knl(*args)),
                cls.covar(*args))

class Affine(GPFunctionGroup):
    """
    return aᵀf, aᵀk_fa, k_fa
    """
    def mean(affine, f):
        return affine.T @ f.mean

    def knl(affine, f):
        return affine.T @ f.k @ affine


    def covar(affine, f):
        return f.k @ affine.T


class GradientGP(GPFunctionGroup):
    """
    return ∇f, Hₓₓ k_f, ∇ covar_fg
    """
    def mean(f, x, covar_fg):
        return torch.autograd.grad(f.mean, x, retain_graph=True)[0]

    def knl(f, x, covar_fg):
        grad_k = torch.autograd.grad(f.k, x, create_graph=True)[0]
        Hxx_k_v_prod = lambda v: torch.autograd.grad(grad_k, x, grad_outputs=v,
                                                    retain_graph=True)[0]
        return Hxx_k_v_prod

    def covar(f, x, covar_fg):
        J_covar_fg = t_jac(covar_fg, x)
        return J_covar_fg


class QuadraticFormOfGP(GPFunctionGroup):
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
    def mean(g, f, covar_gf):
        mean_quad = g.mean @ f.mean + covar_gf.trace()
        return mean_quad

    def knl(g, f, covar_gf):
        k_quad = (2 * covar_gf.trace()
                + 2 * g.mean.T @ covar_gf @ f.mean
                + g.mean.T @ f.k @ f.mean
                + f.mean.T @ g.k(f.mean))
        return k_quad

    def covar(g, f, covar_hg, covar_hf=None):
        """
        Computes cov(h, g*f)

        If covar_hf is None, then computes cov(f, g*f), then covar_hg should be
        cov(f, g)
        """
        if covar_hf is None:
            # this means that h = f
            covar_hf = f.k
        covar_quad_f = covar_hg @ f.mean + covar_hf @ g.mean
        return covar_quad_f


class AddGP(GPFunctionGroup):
    """
    s = f + g
    """
    def mean(f, g, covar_fg):
        return f.mean + g.mean

    def knl(f, g, covar_fg):
        return f.k + g.k + 2*covar_fg

    def covar(f, g, covar_fg):
        return covar_fg


def lie1_gp(h, fu, x):
    """
    L_f h(x) = ∇ h(x)ᵀ f(x; u)
    """
    grad_h = torch.autograd.grad(h, x, retain_graph=True, create_graph=True)[0]
    return Affine.gp(grad_h, fu)


def lie2_gp(h_func, fu_func, x, u):
    """
    L_f² h(x) = ∇[L_f h(x)]ᵀ f(x; u)
    """
    x = x.requires_grad_(True)
    h = h_func(x)
    fu = fu_func(x, u)
    L1h, covar_L1h_fu = lie1_gp(h, fu, x)
    grad_L1h, covar_grad_L1h_fu = GradientGP.gp(L1h, x, covar_L1h_fu)
    covar_grad_L1h_L1h = GradientGP.covar(L1h, x, L1h.k)
    L2h, covar_L2h_fu = QuadraticFormOfGP.gp(grad_L1h, fu, covar_grad_L1h_fu)
    covar_L2h_L1h = QuadraticFormOfGP.covar(grad_L1h, fu,
                                            covar_grad_L1h_L1h,
                                            covar_L1h_fu)
    assert L2h.mean.ndim == 0
    return L2h, covar_L2h_fu, covar_L2h_L1h


def cbc2_gp(h_func, fu_func, x, u, K_α=[1,1]):
    """
    L_f² h(x) + K_α [     h(x) ] ≥ 0
                    [ L_f h(x) ]
    """
    x.requires_grad_(True)
    u.requires_grad_(True)
    L1h, _ = lie1_gp(h_func(x), fu_func(x, u), x)
    L2h, _, covar_L2h_L1h = lie2_gp(h_func, fu_func, x, u)
    h = h_func(x)
    cbc2, _ = AddGP.gp(L2h, GaussianProcess(K_α[1] * L1h.mean + K_α[0] * h,
                                         K_α[1] * K_α[1] * L1h.k),
                                   K_α[1] * covar_L2h_L1h)
    assert cbc2.mean.ndim == 0
    assert cbc2.k.ndim == 0
    return cbc2

def get_affine_terms(func, x):
    x = x.requires_grad_(True)
    f_x = func(x)
    linear = torch.autograd.grad(f_x, x, create_graph=True)[0]
    with torch.no_grad():
        const = f_x - linear @ x
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
    mean_Q = mean_A.T @ mean_A
    mean_p = 2 * mean_A @ mean_b if mean_b.ndim else 2 * mean_A * mean_b
    mean_r = mean_b @ mean_b if mean_b.ndim else mean_b * mean_b
    k_func = lambda up : cbc2_gp(h_func, fu_func, x, up).k
    k_Q, k_p, k_r = get_quadratic_terms(k_func, u)
    ratio = (1-δ) / δ
    return ratio * k_Q - mean_Q, ratio * k_p - mean_p, ratio * k_r - mean_r




