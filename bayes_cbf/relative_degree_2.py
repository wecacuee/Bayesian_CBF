import numpy as np
import torch
from contextlib import contextmanager
from collections import namedtuple
from functools import partial

from bayes_cbf.control_affine_model import GaussianProcess, GaussianProcessFunc
from bayes_cbf.misc import t_jac, variable_required_grad, t_hessian
from importlib import import_module
tgradcheck = import_module("torch.autograd.gradcheck") # global variable conflicts with module name

EPS = 2e-3

class AffineGP:
    """
    return aᵀf, aᵀk_fa, k_fa
    """
    def __init__(self, affine, f):
        self.affine = affine
        self.f = f

    @property
    def dtype(self):
        return self.f.mean.__self__.dtype

    def to(self, dtype):
        self.f.to(dtype=dtype)
        self.affine.__self__.to(dtype=dtype)

    def mean(self, x):
        affine, f = self.affine, self.f
        return affine(x).T @ f.mean(x)

    def knl(self, x, xp):
        affine, f = self.affine, self.f
        var = affine(x).T @ f.knl(x, xp) @ affine(x)
        assert (var >= -EPS).all()
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
    def __init__(self, f, grad_check=False, analytical_hessian=True):
        self.f = f
        self.grad_check = grad_check
        self.analytical_hessian = analytical_hessian

    @property
    def dtype(self):
        return self.f.dtype

    def to(self, dtype):
        self.f.to(dtype)

    def mean(self, x):
        f = self.f

        if self.grad_check:
            old_dtype = self.dtype
            self.to(torch.float64)
            with variable_required_grad(x):
                torch.autograd.gradcheck(f.mean, x.double())
            self.to(dtype=old_dtype)
        with variable_required_grad(x):
            return torch.autograd.grad(f.mean(x), x)[0]

    def knl(self, x, xp, eigeps=EPS):
        f = self.f
        if xp is x:
            xp = xp.detach().clone()

        grad_k_func = lambda xs, xt: torch.autograd.grad(
            f.knl(xs, xt), xs, create_graph=True)[0]
        if self.grad_check:
            old_dtype = self.dtype
            self.to(torch.float64)
            f_knl_func = lambda xt: f.knl(xt, xp.double())
            with variable_required_grad(x):
                torch.autograd.gradcheck(f_knl_func, x.double())
            torch.autograd.gradgradcheck(lambda x: f.knl(x, x), x.double())
            with variable_required_grad(x):
                with variable_required_grad(xp):
                    torch.autograd.gradcheck(
                        lambda xp: grad_k_func(x.double(), xp)[0], xp.double())
            self.to(dtype=old_dtype)

        analytical = self.analytical_hessian
        if analytical:
            Hxx_k = t_hessian(f.knl, x, xp)
        else:
            with variable_required_grad(x):
                with variable_required_grad(xp):
                    old_dtype = self.dtype
                    self.to(torch.float64)
                    Hxx_k = tgradcheck.get_numerical_jacobian(
                        partial(grad_k_func, x.double()), xp.double())
                    self.to(dtype=old_dtype)
                    Hxx_k = Hxx_k.to(old_dtype)
        if torch.allclose(x, xp):
            eigenvalues, eigenvectors = torch.eig(Hxx_k, eigenvectors=False)
            assert (eigenvalues[:, 0] > -eigeps).all(),  " Hessian must be positive definite"
            small_neg_eig = ((eigenvalues[:, 0] > -eigeps) & (eigenvalues[:, 0] < 0))
            if small_neg_eig.any():
                eigenvalues, eigenvectors = torch.eig(Hxx_k, eigenvectors=True)
                evalz = eigenvalues[:, 0]
                evalz[small_neg_eig] = 0
                Hxx_k = eigenvectors.T @ torch.diag(evalz) @ eigenvectors
        return Hxx_k

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
                + g.mean(x).T @ f.knl(x, xp) @ g.mean(xp)
                + f.mean(x).T @ g.knl(x, xp) @ f.mean(xp))
        assert (k_quad >= -EPS).all()
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
        assert (var >= -EPS).all()
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
    def __init__(self, grad_h, f_gp):
        self.grad_h = grad_h
        self.f_gp = f_gp
        self._affine = AffineGP(grad_h, f_gp)

    @property
    def dtype(self):
        return self._affine.dtype

    def to(self, dtype):
        self._affine.to(dtype)

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


class GradLie1GPExplicit:
    """
    ∇ L_f h(x) = ∇ ( ∇ h(x)ᵀ f(x; u) )

    E[ L_f h(x) ] = ∇ ( ∇h(x)ᵀ E[f(x; u)] )
    Var[ L_f h(x) ] = ∇h(x)ᵀ [k(x, x') A ] ∇h(x)
    """
    def __init__(self, h, regressor):
        self.grad_h = grad_operator(h)
        self.regressor = regressor
        self.f = regressor.f_func_gp()

    def _Hxx_h(self, x):
        with variable_required_grad(x):
            return t_jac(self.grad_h(x), x)

    def _Jac_f_mean(self, x):
        # TODO: to commpute
        return self.regressor.custom_predict(x, grad_gp=True, commpute_cov=False)

    def _Hxx_f_knl(self, x, xp):
        # TODO: to compute
        _, scalar_var = self.regressor.custom_predict(x, Xtestp_in=xp, grad_gp=True,
                                                      commpute_cov=True, scalar_var_only=True)
        return scalar_var

    def _grad_f_knl(self, x, xp):
        with variable_required_grad(x):
            return torch.autograd.grad(self.f.knl(x, xp), x)[0]

    def mean(self, x):
        """
        E[ L_f h(x) ] = ∇h(x)ᵀ E[f(x; u)]
        """
        return self._Hxx_h(x) @ self.f.mean(x) + self._Jac_f_mean(x) @ self.grad_h(x)

    def knl(self, x, xp):
        """
        Var[ L_f h(x) ] = ∇h(x)ᵀ E[f(x; u)] ∇h(x)
        """
        grad_f_knl = self._grad_f_knl(x, xp)
        Hxx_f_knl = self._Hxx_f_knl(x, xp)
        grad_hx = self.grad_h
        Hxx_h = self._Hxx_h
        A = self.regressor.A_mat()
        knl = self.f.knl(x, xp)
        return (
            Hxx_f_knl @ grad_hx(x).T @ A @ grad_hx(xp)
            + grad_hx(x) @ A @ Hxx_h(xp) @ grad_f_knl
            + grad_f_knl @ grad_hx(x).T @ A @ Hxx_h(xp)
            + knl * Hxx_h(x).T @ A @ Hxx_h(x)
        )

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
    def __init__(self, lie1_gp, covar_fu_f, fu_gp, caregressor):
        self.fu = fu_gp
        self._lie1_gp = lie1_gp
        self._covar_fu_f = covar_fu_f
        self.covar_Lie1_fu = covar_Lie1_fu = partial(lie1_gp.covar_g, covar_fu_f)
        #self._grad_lie1_gp = GradLie1GPExplicit(lie1_gp.affine, caregressor)
        self._grad_lie1_gp = GradientGP(lie1_gp)
        self.covar_grad_Lie1_fu = covar_grad_Lie1_fu = partial(
            self._grad_lie1_gp.covar_g, covar_Lie1_fu)
        self._quad_gp = QuadraticFormOfGP(self._grad_lie1_gp, fu_gp,
                                          covar_grad_Lie1_fu)

    def mean(self, x):
        return self._quad_gp.mean(x)

    def knl(self, x, xp):
        return self._quad_gp.knl(x, xp)

    def _covars(self):
        covar_L1h_fu = self.covar_Lie1_fu
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


def cbc2_gp(h_func, grad_h_func, learned_model, u, K_α=[1,1]):
    """
    L_f² h(x) + K_α [     h(x) ] ≥ 0
                    [ L_f h(x) ]
    """
    f_gp = learned_model.f_func_gp()
    L1h = Lie1GP(grad_h_func, f_gp)
    covar_fu_f = partial(learned_model.covar_fu_f, u)
    fu_func = learned_model.fu_func_gp(u)
    L2h = Lie2GP(L1h, covar_fu_f, fu_func, learned_model)
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
        linear_more = torch.autograd.grad(f_x, x, create_graph=True)[0]
        quad = t_jac(linear_more, x) / 2
    with torch.no_grad():
        linear = linear_more - 2 * quad @ x
        const = f_x - x.T @ quad @ x - linear @ x
    return quad, linear, const


def cbc2_quadratic_terms(h_func, grad_h_func, control_affine_model, x, u):
    """
    cbc2.mean(x) ≥ √(1-δ)/δ cbc2.k(x,x')
    """
    mean = lambda up: cbc2_gp(h_func, grad_h_func, control_affine_model, up).mean(x)
    k_func = lambda up: cbc2_gp(h_func, grad_h_func, control_affine_model, up).knl(x, x)

    #assert mean(u) > 0, 'cbf2 should be at least satisfied in expectation'
    mean_A, mean_b = get_affine_terms(mean, u)
    assert not torch.isnan(mean_A).any()
    assert not torch.isnan(mean_b).any()
    k_Q, k_p, k_r = get_quadratic_terms(k_func, u)
    assert not torch.isnan(k_Q).any()
    assert not torch.isnan(k_p).any()
    assert not torch.isnan(k_r).any()
    return (mean_A, mean_b), (k_Q, k_p, k_r), mean(u), k_func(u)
