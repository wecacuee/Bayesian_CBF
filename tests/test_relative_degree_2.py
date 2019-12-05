
from bayes_cbf.pendulum import RadialCBFRelDegree2
from bayes_cbf.control_affine_model import GaussianProcess
from bayes_cbf.relative_degree_2 import AffineGP, GradientGP, QuadraticFormOfGP
from .test_control_affine_regression import test_pendulum_train_predict

def test_affine_gp(dgp=None):
    if dgp is None:
        dgp = test_pendulum_train_predict()
    xtest = torch.rand(2)
    utest = torch.rand(1)
    Xtest = xtest.unsqueeze(0)
    cbf2 = RadialCBFRelDegree2(dgp)
    l1h = AffineGP(
        cbf2.grad_h2_col,
        dgp.f_func_gp(utest))
    l1h.mean(xtest)
    l1h.knl(xtest, xtest)
    return l1h

def test_gradient_gp(dgp=None):
    if dgp is None:
        dgp = test_pendulum_train_predict()
    l1h = test_affine_gp(dgp)
    grad_l1h = GradientGP(l1h)
    xtest = torch.rand(2)
    grad_l1h.mean(xtest)
    grad_l1h.knl(xtest, xtest)
    return grad_l1h

def test_quadtratic_form(dgp=None):
    if dgp is None:
        dgp = test_pendulum_train_predict()
    grad_l1h = test_gradient_gp(dgp)
    utest = torch.rand(1)
    l2h = QuadraticFormOfGP(grad_l1h, dgp.f_func_gp(utest))
    xtest = torch.rand(2)
    l2h.mean(xtest)
    l2h.knl(xtest, xtest)
    return l2h

def test_lie2_gp(dgp=None):
    if dgp is None:
        dgp = test_pendulum_train_predict()
    xtest = torch.rand(2)
    Xtest = xtest.unsqueeze(0)
    utest = torch.rand(1)
    Utest = utest.unsqueeze(0)
    cbf2 = RadialCBFRelDegree2(dgp)
    L2h = Lie2GP(Lie1GP(cbf2.h2_col, dgp.f_func_gp(utest)),
                 dgp.f_func_gp(utest))
    L2h.mean(xtest)
    L2h.knl(xtest, xtest)

def test_quadtratic_term():
    Q = torch.eye(2)
    a = torch.tensor([2.0, 3.0])
    c = torch.tensor(4.0)
    y_func = lambda x: x @ Q @ x +  a @ x + c
    x = torch.rand(2)
    Qp, ap, cp = get_quadratic_terms(y_func, x)
    torch.allclose(Q, Qp)
    torch.allclose(a, ap)
    torch.allclose(c, cp)


