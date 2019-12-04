
from bayes_cbf.pendulum import RadialCBFRelDegree2
from bayes_cbf.relative_degree_2 import affine_gp
from bayes_cbf.control_affine_model import test_pendulum_train_predict, GaussianProcess

def test_affine_gp():
    dgp = test_pendulum_train_predict()
    xtest = torch.rand(2)
    Xtest = xtest.unsqueeze(0)
    cbf2 = RadialCBFRelDegree2(dgp)
    xtest.requires_grad_(True)
    l1h = affine_gp(
        cbf2.grad_h2_col(xtest),
        dgp.f_func_gp(xtest))
    grad_l1h = grad_gp(l1h, xtest)
    l2h = quadratic_form_of_gps(grad_l1h, dgp.f_func_gp(xtest))

def test_lie2_gp():
    dgp = test_pendulum_train_predict()
    xtest = torch.rand(2)
    Xtest = xtest.unsqueeze(0)
    utest = torch.rand(1)
    Utest = utest.unsqueeze(0)
    cbf2 = RadialCBFRelDegree2(dgp)
    xtest.requires_grad_(True)
    utest.requires_grad_(True)
    lie2_gp(cbf2.h2_col, dgp.f_func_gp, xtest, utest)

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


