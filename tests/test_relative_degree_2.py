from functools import partial

import torch

from bayes_cbf.pendulum import RadialCBFRelDegree2
from bayes_cbf.control_affine_model import GaussianProcess
from bayes_cbf.relative_degree_2 import (AffineGP, GradientGP,
                                         QuadraticFormOfGP, Lie1GP, Lie2GP,
                                         cbc2_gp, get_quadratic_terms,
                                         cbc2_quadratic_terms)
from bayes_cbf.misc import to_numpy
from tests.test_control_affine_regression import test_pendulum_train_predict

import pytest

_global_cache = None
@pytest.fixture
def dynamic_models():
    global _global_cache
    if _global_cache is None:
        _global_cache = test_pendulum_train_predict()
    return _global_cache


def get_close_xtest_sample(learned_model):
    MXUHtrain = learned_model.model.train_inputs[0]
    train_size = MXUHtrain.shape[0]
    xtrain = MXUHtrain[torch.randint(train_size, ()), 1:3].cpu()
    xtest = xtrain + 1e-3 * torch.rand_like(xtrain)
    return xtest


def test_affine_gp(dynamic_models):
    learned_model, true_model = dynamic_models
    true_cbf2 = RadialCBFRelDegree2(true_model)
    learned_cbf2 = RadialCBFRelDegree2(learned_model)
    xtest = get_close_xtest_sample(learned_model)
    l1h = AffineGP(
        learned_cbf2.grad_h2_col,
        learned_model.f_func_gp())
    assert to_numpy(l1h.mean(xtest)) == pytest.approx(
        to_numpy(true_cbf2.lie_f_h2_col(xtest)), rel=0.1)
    l1h.knl(xtest, xtest)
    return l1h


def test_gradient_gp(dynamic_models):
    learned_model, true_model = dynamic_models
    l1h = test_affine_gp(dynamic_models)
    true_cbf2 = RadialCBFRelDegree2(true_model)
    grad_l1h = GradientGP(l1h)
    xtest = get_close_xtest_sample(learned_model)
    assert to_numpy(grad_l1h.mean(xtest)) == pytest.approx(to_numpy(true_cbf2.grad_lie_f_h2_col(xtest)), abs=0.1, rel=0.1)
    grad_l1h.knl(xtest, xtest)
    return grad_l1h, l1h


def test_quadratic_form(dynamic_models):
    learned_model, true_model = dynamic_models
    true_cbf2 = RadialCBFRelDegree2(true_model)
    grad_l1h, l1h = test_gradient_gp(dynamic_models)
    utest = torch.rand(1)
    covar_fu_f = partial(learned_model.covar_fu_f, utest)
    covar_Lie1_fu = partial(l1h.covar_g, covar_fu_f)
    covar_grad_l1h_fu = partial(grad_l1h.covar_g, covar_Lie1_fu)
    l2h = QuadraticFormOfGP(grad_l1h, learned_model.fu_func_gp(utest),
                            covar_grad_l1h_fu)
    xtest = get_close_xtest_sample(learned_model)
    assert to_numpy(l2h.mean(xtest)) == pytest.approx(
        to_numpy(
            true_cbf2.lie2_f_h_col(xtest)
            + true_cbf2.lie_g_lie_f_h_col(xtest) * utest
        ), abs=0.1, rel=0.1)
    l2h.knl(xtest, xtest)
    return l2h


def test_lie2_gp(dynamic_models):
    learned_model, true_model = dynamic_models
    true_cbf2 = RadialCBFRelDegree2(true_model)
    xtest = get_close_xtest_sample(learned_model)
    utest = torch.rand(1)
    cbf2 = RadialCBFRelDegree2(dynamic_models)
    f_gp = learned_model.f_func_gp()
    covar_fu_f = partial(learned_model.covar_fu_f, utest)
    fu_gp = learned_model.fu_func_gp(utest)
    L2h = Lie2GP(Lie1GP(cbf2.h2_col, f_gp), covar_fu_f, fu_gp)
    L2h.mean(xtest)
    L2h.knl(xtest, xtest)
    return L2h


def test_cbf2_gp(dynamic_models):
    learned_model, true_model = dynamic_models
    true_cbf2 = RadialCBFRelDegree2(true_model)
    learned_cbf2 = RadialCBFRelDegree2(learned_model)
    utest = torch.rand(1) * 2 - 1
    cbc2 = cbc2_gp(learned_cbf2.h2_col, learned_model, utest)
    xtest = get_close_xtest_sample(learned_model)
    assert to_numpy(cbc2.mean(xtest)) == pytest.approx(to_numpy(
        - true_cbf2.A(xtest) @ utest + true_cbf2.b(xtest))[0], rel=0.1)
    cbc2.knl(xtest, xtest)


def test_cbc2_quadtratic_terms(dynamic_models):
    learned_model, true_model = dynamic_models
    true_cbf2 = RadialCBFRelDegree2(true_model)
    learned_cbf2 = RadialCBFRelDegree2(learned_model)
    utest = torch.rand(1) * 2 - 1
    xtest = get_close_xtest_sample(learned_model)
    (mean_A, mean_b), _ = cbc2_quadratic_terms(
        learned_cbf2.h2_col, learned_model, xtest, utest)


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


