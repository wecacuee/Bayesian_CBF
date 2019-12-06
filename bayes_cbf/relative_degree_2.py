import numpy as np
import torch
from contextlib import contextmanager
from collections import namedtuple
from functools import partial

from bayes_cbf.control_affine_model import GaussianProcess, GaussianProcessFunc
from bayes_cbf.misc import t_jac, variable_required_grad

class AffineGP:
    """
    return aᵀf, aᵀk_fa, k_fa
    """
    def __init__(self, affine, f):
        self.affine = affine
        self.f = f

    def mean(self, x):
        affine, f = self.affine, self.f
        return affine(x).T @ f.mean(x)

    def knl(self, x, xp):
        affine, f = self.affine, self.f
        var = affine(x).T @ f.knl(x, xp) @ affine(x)
        assert (var >= -1e-3).all()
        return var

    def covar_f(self, x, xp):
        affine, f = self.affine, self.f
        return f.knl(x, xp) @ affine(xp).T

    def covar_g(self, cov_fg, x, xp):
        """
        Return cov(affineᵀf, g) for any g such that cov(f, g) is given
        """
        affine, f = self.affine, self.f
        return affine(x).T @ cov_fg(x, xp)


class GradientGP:
    """
    return ∇f, Hₓₓ k_f, ∇ covar_fg
    """
    def __init__(self, f):
        self.f = f

    def mean(self, x):
        f = self.f
        with variable_required_grad(x):
            return torch.autograd.grad(f.mean(x), x)[0]

    def knl(self, x, xp):
        f = self.f
        with variable_required_grad(x):
            with variable_required_grad(xp):
                grad_k = torch.autograd.grad(f.knl(x, xp), x, create_graph=True)[0]
                Hxx_k = t_jac(grad_k, xp)
        eigenvalues, _ = torch.eig(Hxx_k)
        assert (eigenvalues[:, 0] > -1e-3).all()
        return Hxx_k # must be positive definite

    def covar_g(self, covar_fg_func, x, xp):
        """
        returns cov(∇f, g) given cov(f, g)
        """
        f = self.f
        with variable_required_grad(x):
            J_covar_fg = t_jac(covar_fg_func(x, xp), x)
        return J_covar_fg


class QuadraticFormOfGP:
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
    def __init__(self, g, f, covar_gf):
        self.g = g
        self.f = f
        self.covar_gf = covar_gf

    def mean(self, x):
        g, f, covar_gf = self.g, self.f, self.covar_gf
        mean_quad = g.mean(x) @ f.mean(x) + covar_gf(x, x).trace()
        return mean_quad

    def knl(self, x, xp):
        g, f, covar_gf = self.g, self.f, self.covar_gf
        k_quad = (2 * covar_gf(x, xp).trace()**2
                + g.mean(x).T @ covar_gf(x, xp) @ f.mean(xp) * 2
                + g.mean(x).T @ f.knl(x, xp) @ f.mean(xp)
                + f.mean(x).T @ g.knl(x, xp) @ f.mean(xp))
        assert (k_quad >= -1e-3).all()
        return k_quad

    def covar_h(self, covar_hf, covar_hg, x, xp):
        """
        Computes cov(h, g*f) given cov(h, f) and cov(h, g)

        If covar_hf is None, then computes cov(f, g*f), then covar_hg should be
        cov(f, g)
        """
        g, f, covar_gf = self.g, self.f, self.covar_gf
        if covar_hf is None:
            # this means that h = f
            covar_hf = f.knl
        covar_quad_f = covar_hg(x, xp) @ f.mean(xp) + covar_hf(x, xp) @ g.mean(xp)
        return covar_quad_f


class AddGP:
    """
    s = f + g
    """
    def __init__(self, f, g, covar_fg):
        self.f = f
        self.g = g
        self.covar_fg = covar_fg

    def mean(self, x):
        f, g, covar_fg = self.f, self.g, self.covar_fg
        return f.mean(x) + g.mean(x)

    def knl(self, x, xp):
        f, g, covar_fg = self.f, self.g, self.covar_fg
        var = f.knl(x, xp) + g.knl(x, xp) + 2*covar_fg(x, xp)
        assert (var >= -1e-3).all()
        return var

    def covar_f(self, x, xp):
        f, g, covar_fg = self.f, self.g, self.covar_fg
        return f.knl(x, xp) + covar_fg(x, xp)

    def covar_g(self, x, xp):
        f, g, covar_fg = self.f, self.g, self.covar_fg
        return g.knl(x, xp) + covar_fg(x, xp)


def grad_operator(func):
    def grad_func(x):
        with variable_required_grad(x):
            return torch.autograd.grad(func(x), x, create_graph=True)[0]
    return grad_func


class Lie1GP:
    """
    L_f h(x) = ∇ h(x)ᵀ f(x; u)
    """
    def __init__(self, h, f_gp):
        self.h = h
        self.f_gp = f_gp
        self._affine = AffineGP(grad_operator(h), f_gp)

    def mean(self, x):
        return self._affine.mean(x)

    def knl(self, x, xp):
        return self._affine.knl(x, xp)

    def covar_f(self, x, xp):
        return self._affine.covar_f(x, xp)

    def covar_g(self, cov_fg, x, xp):
        """
        Return cov(∇hᵀf, g) for any g such that cov(f, g) is given
        """
        return self._affine.covar_g(cov_fg, x, xp)


class Lie2GP:
    """
    L_f² h(x) = ∇[L_f h(x)]ᵀ f(x; u)
    """
    def __init__(self, lie1_gp, covar_fu_f, fu_gp):
        self.fu = fu_gp
        self._lie1_gp = lie1_gp
        covar_Lie1_fu = partial(lie1_gp.covar_g, covar_fu_f)
        self._grad_lie1_gp = GradientGP(lie1_gp)
        covar_grad_Lie1_fu = partial(self._grad_lie1_gp.covar_g,
                                     covar_Lie1_fu)
        self._quad_gp = QuadraticFormOfGP(self._grad_lie1_gp, fu_gp,
                                          covar_grad_Lie1_fu)

    def mean(self, x):
        return self._quad_gp.mean(x)

    def knl(self, x, xp):
        return self._quad_gp.knl(x, xp)

    def _covars(self):
        covar_L1h_fu = self._lie1_gp.covar_fu
        covar_grad_L1h_fu = partial(self._grad_lie1_gp.covar_g, covar_L1h_fu)
        covar_grad_L1h_L1h = partial(self._grad_lie1_gp.covar_g, self._lie1_gp.knl)
        covar_L2h_fu = partial(self._quad_gp.covar_h, covar_grad_L1h_fu, self.fu.knl)
        covar_L2h_L1h = partial(self._quad_gp.covar_h, covar_grad_L1h_L1h, covar_L1h_fu)
        return covar_L2h_fu, covar_L2h_L1h

    def covar_fu(self, x, xp):
        covar_L2h_fu, _ = self._covars()
        return covar_L2h_fu(x, xp)

    def covar_lie1(self, x, xp):
        _, covar_L2h_L1h = self._covars()
        return covar_L2h_L1h(x, xp)


def cbc2_gp(h_func, learned_model, u, K_α=[1,1]):
    """
    L_f² h(x) + K_α [     h(x) ] ≥ 0
                    [ L_f h(x) ]
    """
    f_gp = learned_model.f_func_gp()
    L1h = Lie1GP(h_func, f_gp)
    covar_fu_f = partial(learned_model.covar_fu_f, u)
    fu_func = learned_model.fu_func_gp(u)
    L2h = Lie2GP(L1h, covar_fu_f, fu_func)
    cbc2 = AddGP(L2h,
                 GaussianProcessFunc(
                     mean=lambda x: K_α[1] * L1h.mean(x) + K_α[0] * h_func(x),
                     knl=lambda x, xp: K_α[1] * K_α[1] * L1h.knl(x, xp)),
                 lambda x, xp: K_α[1] * L2h.covar_lie1(x, xp))
    return cbc2


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
        linear = torch.autograd.grad(f_x, x, create_graph=True)[0]
        quad = t_jac(linear_more, x) / 2
    with torch.no_grad():
        linear = linear_more - 2 * quad @ x
        const = f_x - x.T @ quad @ x - linear @ x
    return quad, linear, const


def cbf2_quadratic_constraint(h_func, control_affine_model, x, u):
    """
    cbc2.mean(x) ≥ √(1-δ)/δ cbc2.k(x,x')
    """
    mean = lambda up: cbc2_gp(h_func, control_affine_model, up).mean(x)
    k_func = lambda up: cbc2_gp(h_func, control_affine_model, up).knl(
        x, x.detach().clone())

    assert mean(u) > 0, 'cbf2 should be at least satisfied in expectation'
    mean_A, mean_b = get_affine_terms(mean, u)
    k_Q, k_p, k_r = get_quadratic_terms(k_func, u)
    #ratio = (1-δ) / δ
    return (mean_A, mean_b), (k_Q, k_p, k_r)

