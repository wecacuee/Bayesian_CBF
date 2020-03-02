from functools import partial
import os.path as osp

import torch

from gpytorch.kernels import RBFKernel
from gpytorch.kernels.kernel import Distance, default_postprocess_script
import gpytorch.settings

from bayes_cbf.pendulum import (RadialCBFRelDegree2, PendulumDynamicsModel,
                                ControlAffineRegressor)
from bayes_cbf.control_affine_model import GaussianProcess
from bayes_cbf.relative_degree_2 import (AffineGP, GradientGP,
                                         QuadraticFormOfGP, Lie1GP, Lie2GP,
                                         cbc2_gp, get_quadratic_terms,
                                         cbc2_quadratic_terms)
from bayes_cbf.misc import to_numpy, variable_required_grad
from tests.test_control_affine_regression import test_pendulum_train_predict

import pytest


def get_test_sample_close_to_train(learned_model, dist=0.1):
    MXUHtrain = learned_model.model.train_inputs[0]
    train_size = MXUHtrain.shape[0]
    idx = torch.randint(train_size-1, ())
    xtrain_utrain = MXUHtrain[idx, 1:].cpu()
    xtest = xtrain_utrain[:2]
    xtest = xtest + dist * torch.norm(xtest) * torch.rand_like(xtest)
    utest = xtrain_utrain[3:]
    utest = utest + dist * torch.norm(utest) * torch.rand_like(utest)
    return xtest, utest


def rel2abs(f, basedir=osp.dirname(__file__) or "."):
    return osp.join(basedir, f)


_global_cache = None
@pytest.fixture
def dynamic_models(learned_model_path='data/pendulum_learned_model.torch'):
    global _global_cache
    if _global_cache is None:
        if True or not osp.exists(rel2abs(learned_model_path)):
            learned_model, true_model = test_pendulum_train_predict()
            torch.save(learned_model.state_dict(), rel2abs(learned_model_path))
        else:
            true_model = PendulumDynamicsModel(n=2, m=1)
            learned_model = ControlAffineRegressor(2, 1)
            learned_model.load_state_dict(
                torch.load(rel2abs(learned_model_path)))
        xtest, utest = get_test_sample_close_to_train(learned_model)
        _global_cache = learned_model, true_model, xtest, utest
    return _global_cache


def test_affine_gp(dynamic_models, skip_test=False):
    learned_model, true_model, xtest, _ = dynamic_models
    true_cbf2 = RadialCBFRelDegree2(true_model)
    learned_cbf2 = RadialCBFRelDegree2(learned_model)
    l1h = AffineGP(
        learned_cbf2.grad_h2_col,
        learned_model.f_func_gp())
    if not skip_test:
        assert to_numpy(l1h.mean(xtest)) == pytest.approx(
            to_numpy(true_cbf2.lie_f_h2_col(xtest)), rel=0.1)
    l1h.knl(xtest, xtest)
    return l1h


class SimpleGP:
    def __init__(self, m, lengthscale):
        self.m = m
        self.lengthscale = lengthscale

    @property
    def dtype(self):
        return self.lengthscale.dtype

    def to(self, dtype):
        self.m = self.m.to(dtype=dtype)
        self.lengthscale = self.lengthscale.to(dtype=dtype)

    def mean(self, x):
        m = self.m
        return m @ x

    def grad_mean(self, x):
        m = self.m
        return m

    def knl(self, x, xp):
        lengthscale = self.lengthscale
        diff = (x - xp) / lengthscale
        return (diff.T @ diff).div_(-2.0).exp_()

    def knl_grad(self, x, xp):
        lengthscale = self.lengthscale
        diff = (x - xp) / lengthscale
        return -diff * (diff.T @ diff).div_(-2.0).exp_() / lengthscale

    def knl_hessian(self, x, xp):
        lengthscale = self.lengthscale
        diff = (x - xp) / lengthscale**2
        H_xx_k_1 =  (diff.unsqueeze(-1) @ diff.unsqueeze(0)) * self.knl(x, xp)
        H_xx_k_2 = torch.eye(x.shape[0]) / lengthscale**2 * self.knl(x, xp)
        return H_xx_k_2 - H_xx_k_1


def test_gradient_simple():
    m = torch.rand(2)
    lengthscale = torch.rand(2)
    simp_gp = SimpleGP(m, lengthscale)
    grad_simp_gp = GradientGP(simp_gp, analytical_hessian=True)
    xtest = torch.rand(2)
    assert to_numpy(simp_gp.grad_mean(xtest)) == pytest.approx(
        to_numpy(grad_simp_gp.mean(xtest)))
    assert to_numpy(simp_gp.knl_hessian(xtest, xtest)) == pytest.approx(
        to_numpy(grad_simp_gp.knl(xtest, xtest)))
    xtestp = torch.rand(2)
    assert to_numpy(simp_gp.knl_hessian(xtest, xtestp)) == pytest.approx(
        to_numpy(grad_simp_gp.knl(xtest, xtestp)), rel=1e-3, abs=1e-5)


def test_gradient_f_gp(dynamic_models, skip_test=False, dist=1e-4):
    learned_model, true_model, xtest, utest = dynamic_models
    grad_f = GradientGP(AffineGP(
        lambda x: torch.tensor([1., 0.]),
        learned_model.f_func_gp()))
    def xdot_func(x):
        return true_model.f_func(x)[0] + (true_model.g_func(x) @ utest)[0]
    with variable_required_grad(xtest):
        true_grad_f = torch.autograd.grad(xdot_func(xtest), xtest)[0]
    if not skip_test:
        assert to_numpy(grad_f.mean(xtest)) == pytest.approx(to_numpy(true_grad_f), abs=0.1, rel=0.4)
    grad_f.knl(xtest, xtest)
    return grad_f


#@pytest.mark.skip(reason="Does not succeed in theta")
def test_gradient_gp(dynamic_models, skip_test=False, dist=1e-4, grad_check=True):
    learned_model, true_model, xtest, _ = dynamic_models
    if grad_check:
        learned_model.double_()
        func = lambda lm, X: lm.f_func_knl(X, xtest.double())[0,0]
        with variable_required_grad(xtest):
            torch.autograd.gradcheck(partial(func, learned_model), xtest.double())
        learned_model.float_()
    l1h = test_affine_gp(dynamic_models, skip_test=True)
    true_cbf2 = RadialCBFRelDegree2(true_model)
    grad_l1h = GradientGP(l1h)
    if not skip_test:
        assert to_numpy(grad_l1h.mean(xtest)) == pytest.approx(to_numpy(true_cbf2.grad_lie_f_h2_col(xtest)), abs=0.1, rel=0.4)
    grad_l1h.knl(xtest, xtest)
    return grad_l1h, l1h


#@pytest.mark.skip(reason="Almost always fails")
def test_quadratic_form(dynamic_models, skip_test=False, dist=1e-4):
    learned_model, true_model, xtest, utest = dynamic_models
    true_cbf2 = RadialCBFRelDegree2(true_model)
    grad_l1h, l1h = test_gradient_gp(dynamic_models, skip_test=True)

    covar_fu_f = partial(learned_model.covar_fu_f, utest)
    covar_Lie1_fu = partial(l1h.covar_g, covar_fu_f)
    covar_grad_l1h_fu = partial(grad_l1h.covar_g, covar_Lie1_fu)
    l2h = QuadraticFormOfGP(grad_l1h, learned_model.fu_func_gp(utest),
                            covar_grad_l1h_fu)
    if not skip_test:
        assert to_numpy(l2h.mean(xtest)) == pytest.approx(
            to_numpy(
                true_cbf2.lie2_f_h_col(xtest)
                + true_cbf2.lie_g_lie_f_h_col(xtest) * utest
            )[0], abs=0.1, rel=0.4)
    l2h.knl(xtest, xtest)
    return l2h


def test_lie2_gp(dynamic_models):
    learned_model, true_model, xtest, utest = dynamic_models
    true_cbf2 = RadialCBFRelDegree2(true_model)
    cbf2 = RadialCBFRelDegree2(learned_model)

    f_gp = learned_model.f_func_gp()
    covar_fu_f = partial(learned_model.covar_fu_f, utest)
    fu_gp = learned_model.fu_func_gp(utest)
    L2h = Lie2GP(Lie1GP(cbf2.grad_h2_col, f_gp), covar_fu_f, fu_gp, learned_model)
    assert to_numpy(L2h.mean(xtest)) == pytest.approx(
        to_numpy(
            true_cbf2.lie2_f_h_col(xtest)
            + true_cbf2.lie_g_lie_f_h_col(xtest) * utest
        )[0], abs=0.1, rel=0.4)
    L2h.knl(xtest, xtest)
    return L2h


def test_cbf2_gp(dynamic_models):
    learned_model, true_model, xtest, utest = dynamic_models
    true_cbf2 = RadialCBFRelDegree2(true_model)
    learned_cbf2 = RadialCBFRelDegree2(learned_model)
    cbc2 = cbc2_gp(learned_cbf2.h2_col,
                   learned_cbf2.grad_h2_col, learned_model, utest)
    assert to_numpy(cbc2.mean(xtest)) == pytest.approx(to_numpy(
        - true_cbf2.A(xtest) @ utest + true_cbf2.b(xtest)), rel=0.1, abs=0.1)
    cbc2.knl(xtest, xtest)


def test_cbc2_quadtratic_terms(dynamic_models):
    learned_model, true_model, xtest, utest = dynamic_models
    true_cbf2 = RadialCBFRelDegree2(true_model)
    learned_cbf2 = RadialCBFRelDegree2(learned_model)
    (mean_A, mean_b), knl_terms, mean, knl = cbc2_quadratic_terms(
        learned_cbf2.h2_col, learned_cbf2.grad_h2_col, learned_model, xtest, utest)
    assert to_numpy(mean_A @ utest + mean_b) == pytest.approx(to_numpy(
        - true_cbf2.A(xtest) @ utest + true_cbf2.b(xtest)), rel=0.1)


def test_quadratic_term():
    Q = torch.eye(2)
    a = torch.tensor([2.0, 3.0])
    c = torch.tensor(4.0)
    y_func = lambda x: x @ Q @ x +  a @ x + c
    x = torch.rand(2)
    Qp, ap, cp = get_quadratic_terms(y_func, x)
    torch.allclose(Q, Qp)
    torch.allclose(a, ap)
    torch.allclose(c, cp)


def gpytorch_distance_func(x1, x2):
    return Distance().to(dtype=torch.float64)._sq_dist(x1, x2, False, False)


def dist_func_unsqueeze(x1, x2):
    diff = (x1.unsqueeze(-2) - x2.unsqueeze(-3))
    return diff.pow(2).sum(dim=-1)


def dist_func_matmul(x1, x2):
    return (
        ((-2*x1).matmul(x2.transpose(-2, -1)))
        + x1.pow(2).sum(dim=-1, keepdim=True)
        + x2.pow(2).sum(dim=-1).unsqueeze(-2)
    )


def test_hessian_sq_distance(dist_fun=gpytorch_distance_func):
    """
    d(x₁, x₂)     = |x₁ - x₂|²


    ∂d(x₁, x₂)
    —————————     = 2(x₁ - x₂)
    ∂x₁

    ∂²d(x₁, x₂)
    ——————————–   = -2 I
    ∂x₂ ∂x₁
    """
    #dist_fun = torch.cdist
    with gpytorch.settings.lazily_evaluate_kernels(False):
        x = torch.rand(1, 2, dtype=torch.float64)
        xp = x.clone().detach()
        x.requires_grad_(True)
        xp.requires_grad_(True)

        grad_dist_ne = lambda xq: torch.autograd.grad(
            dist_fun(x, xq), x, create_graph=True)[0][0, 0]
        torch.autograd.gradcheck(grad_dist_ne, xp)


test_unsqueeze_dist_func = partial(test_hessian_sq_distance,
                                   dist_fun=dist_func_unsqueeze)


test_matmul_dist_func = partial(test_hessian_sq_distance,
                                dist_fun=dist_func_matmul)


def test_rbf_kernel():
    """
    k(x₁, x₂)   = exp(-|x₁ - x₂|²/l)


    ∂k(x₁, x₂)
    —————————   = -(2/l)(x₁ - x₂)exp(-|x₁ - x₂|²/l)
    ∂x₁

    ∂²d(x₁, x₂)
    —————————   = (2/l) (I  - (2/l)|x₁ - x₂|²) exp(-|x₁ - x₂|²/l)
    ∂x₂ ∂x₁


    ∂²d(x₁, x₂)|
    ——————————–|           = (2/l) (I) exp(-|x₁ - x₂|²/l)
    ∂x₂ ∂x₁    |x₁ = x₂
    """
    knl = RBFKernel().to(dtype=torch.float64)
    with gpytorch.settings.lazily_evaluate_kernels(False):
        x = torch.rand(1, 2, dtype=torch.float64)
        xp = x.clone().detach()
        x.requires_grad_(True)
        xp.requires_grad_(True)
        grad_dist = lambda xq: torch.autograd.grad(
            knl(x, xq), x, create_graph=True)[0][0, 0]
        torch.autograd.gradcheck(grad_dist, xp)

        grad_dist_ne = lambda xq: torch.autograd.grad(
            knl(x, xq), x, create_graph=True)[0][0, 0]
        torch.autograd.gradcheck(grad_dist_ne, xp)

def test_distances():
    x1 = torch.rand(20, 10)
    x2 = torch.rand(30, 10)
    assert torch.allclose(Distance()._sq_dist(x1, x2, False, False),
                          dist_func_unsqueeze(x1, x2))
    assert torch.allclose(Distance()._sq_dist(x1, x2, False, False),
                          dist_func_matmul(x1, x2))


if __name__ == '__main__':
    test_quadratic_term()
    test_gradient_simple()
    learned_model, dynamic_model = test_pendulum_train_predict()
    xtest, utest = get_test_sample_close_to_train(learned_model)
    fixture = (learned_model, dynamic_model, xtest, utest)
    test_affine_gp(fixture)
    test_gradient_gp(fixture)
    #test_gradient_f_gp(fixture)
    test_quadratic_form(fixture)
    test_lie2_gp(fixture)
    test_cbf2_gp(fixture)
    test_cbc2_quadtratic_terms(fixture)
