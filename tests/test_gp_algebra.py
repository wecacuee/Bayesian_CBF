from functools import partial
import os.path as osp

import torch

from gpytorch.kernels import RBFKernel
from gpytorch.kernels.kernel import Distance, default_postprocess_script
import gpytorch.settings

from bayes_cbf.pendulum import (RadialCBFRelDegree2, PendulumDynamicsModel,
                                ControlAffineRegressor)
from bayes_cbf.control_affine_model import GaussianProcess
# from bayes_cbf.relative_degree_2 import (AffineGP, GradientGP,
#                                          QuadraticFormOfGP, Lie1GP, Lie2GP,
#                                          cbc2_gp, get_quadratic_terms,
#                                          cbc2_quadratic_terms)
from bayes_cbf.misc import to_numpy, variable_required_grad
from bayes_cbf.gp_algebra import DeterministicGP, GradientGP
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
    f_gp = learned_model.f_func_gp()
    l1h_v2 = DeterministicGP(learned_cbf2.grad_h2_col, xtest.shape, name="∇ h(x)").t() @ f_gp
    if not skip_test:
        assert to_numpy(l1h_v2.mean(xtest)) == pytest.approx(
            to_numpy(true_cbf2.lie_f_h2_col(xtest)), rel=0.1)

    return l1h_v2


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
    grad_simp_gp = GradientGP(simp_gp, x_shape=(2,), analytical_hessian=True)
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
    grad_f = GradientGP(DeterministicGP(
        lambda x: torch.tensor([1., 0.]), shape=(2,), name="[1, 0]").t()
                        @ learned_model.f_func_gp(),
                        x_shape=(2,))
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
    grad_l1h = GradientGP(l1h, x_shape=xtest.shape)
    if not skip_test:
        assert to_numpy(grad_l1h.mean(xtest)) == pytest.approx(to_numpy(true_cbf2.grad_lie_f_h2_col(xtest)), abs=0.1, rel=0.4)
    grad_l1h.knl(xtest, xtest)
    return grad_l1h, l1h


#@pytest.mark.skip(reason="Almost always fails")
def test_quadratic_form(dynamic_models, skip_test=False, dist=1e-4):
    learned_model, true_model, xtest, utest = dynamic_models
    true_cbf2 = RadialCBFRelDegree2(true_model)
    grad_l1h, l1h = test_gradient_gp(dynamic_models, skip_test=True)

    # covar_fu_f = partial(learned_model.covar_fu_f, utest)
    # covar_Lie1_fu = partial(l1h.covar, learned_model.fu_func_gp(utest))
    covar_grad_l1h_fu = partial(grad_l1h.covar, learned_model.fu_func_gp(utest))
    fu_gp = learned_model.fu_func_gp(utest)
    l2h = grad_l1h.t() @ fu_gp
    if not skip_test:
        assert to_numpy(l2h.mean(xtest)) == pytest.approx(
            to_numpy(
                true_cbf2.lie2_f_h_col(xtest)
                + true_cbf2.lie_g_lie_f_h_col(xtest) * utest
            )[0], abs=0.1, rel=0.4)
    l2h.knl(xtest, xtest)

    l2h_v2 = grad_l1h.t() @ fu_gp
    if not skip_test:
        assert to_numpy(l2h_v2.mean(xtest)) == pytest.approx(
            to_numpy(
                true_cbf2.lie2_f_h_col(xtest)
                + true_cbf2.lie_g_lie_f_h_col(xtest) * utest
            )[0], abs=0.1, rel=0.4)
        assert to_numpy(l2h.knl(xtest, xtest)) == pytest.approx(to_numpy(l2h_v2.knl(xtest, xtest)))
        assert to_numpy(l2h.covar(fu_gp, xtest, xtest)) == pytest.approx(to_numpy(l2h_v2.covar(fu_gp, xtest, xtest)))
    return l2h


def test_lie2_gp(dynamic_models):
    learned_model, true_model, xtest, utest = dynamic_models
    true_cbf2 = RadialCBFRelDegree2(true_model)
    cbf2 = RadialCBFRelDegree2(learned_model)

    f_gp = learned_model.f_func_gp()
    fu_gp = learned_model.fu_func_gp(utest)
    L2h = GradientGP(
        DeterministicGP(cbf2.grad_h2_col, shape=xtest.shape, name="∇ h(x)").t() @ f_gp,
        x_shape=xtest.shape).t() @ fu_gp
    assert to_numpy(L2h.mean(xtest)) == pytest.approx(
        to_numpy(
            true_cbf2.lie2_f_h_col(xtest)
            + true_cbf2.lie_g_lie_f_h_col(xtest) * utest
        )[0], abs=0.1, rel=0.4)
    L2h.knl(xtest, xtest)
    return L2h

def cbc2_gp(h, grad_h, learned_model, utest, K_α):
    f_gp = learned_model.f_func_gp()
    fu_gp = learned_model.fu_func_gp(utest)
    h_gp = DeterministicGP(h, shape=(1,), name="h(x)")
    grad_h_gp = DeterministicGP(grad_h, shape=(learned_model.state_size,), name="∇ h(x)")
    L1h = grad_h_gp.t() @ f_gp
    L2h = GradientGP(L1h, x_shape=(learned_model.state_size,)).t() @ fu_gp
    return L2h + h_gp * K_α[0] + L1h * K_α[1]


def test_cbf2_gp(dynamic_models):
    learned_model, true_model, xtest, utest = dynamic_models
    true_cbf2 = RadialCBFRelDegree2(true_model)
    learned_cbf2 = RadialCBFRelDegree2(learned_model)
    cbc2 = cbc2_gp(learned_cbf2.h2_col,
                   learned_cbf2.grad_h2_col, learned_model, utest,
                   K_α=learned_cbf2.cbf_col_K_alpha)
    assert to_numpy(cbc2.mean(xtest)) == pytest.approx(to_numpy(
        - true_cbf2.A(xtest) @ utest + true_cbf2.b(xtest)), rel=0.1, abs=0.1)
    cbc2.knl(xtest, xtest)




if __name__ == '__main__':
    test_gradient_simple()
    learned_model, dynamic_model = test_pendulum_train_predict()
    xtest, utest = get_test_sample_close_to_train(learned_model)
    fixture = (learned_model, dynamic_model, xtest, utest)
    test_affine_gp(fixture)
    test_gradient_gp(fixture)
    test_quadratic_form(fixture)
    test_lie2_gp(fixture)
    test_cbf2_gp(fixture)
    #test_gradient_f_gp(fixture)
