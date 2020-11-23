import pytest
import torch
from bayes_cbf.controllers import SOCPController, SumDynamicModels
from bayes_cbf.planner import PiecewiseLinearPlanner
from bayes_cbf.misc import to_numpy, random_psd
from bayes_cbf.unicycle import RelDeg1CLF, ShiftInvariantModel, UnicycleDynamicsModel
from bayes_cbf.cbc2 import cbc2_quadratic_terms

def test_convert_cbc_terms_to_socp_term(m = 2,
                                        extravars = 2):
    bfe = torch.rand((m,))
    e = torch.rand(1)
    V_hom = random_psd(m+1)
    V = V_hom[1:, 1:]
    bfv = V_hom[1:, 0] * 2
    v = V_hom[0, 0]
    u = torch.rand((m,))
    A, bfb, bfc, d = SOCPController.convert_cbc_terms_to_socp_terms(
        bfe, e, V, bfv, v, extravars, testing=True)
    y_u  = torch.cat((torch.zeros(extravars), u))
    std_rhs = (A @ y_u + bfb).norm()
    mean_rhs = bfc @ y_u + d
    std_lhs = torch.sqrt(u.T @ V @ u + bfv @ u + v)
    mean_lhs = bfe @ u + e
    assert to_numpy(mean_lhs ) == pytest.approx(to_numpy(mean_rhs ), abs=1e-4, rel=1e-2)
    assert to_numpy(std_lhs ) == pytest.approx(to_numpy(std_rhs ), abs=1e-4, rel=1e-2)
    assert to_numpy(mean_lhs + std_lhs) == pytest.approx(to_numpy(mean_rhs + std_rhs), abs=1e-4, rel=1e-2)

def test_cbc2_quadratic_terms():
    m = 2
    n = 3
    x_d = torch.rand(n)
    u0 = torch.rand(m)
    x = torch.rand(n)
    net_model = SumDynamicModels(
        ShiftInvariantModel(
            n, m),
        UnicycleDynamicsModel())
    clf = RelDeg1CLF(net_model,
                     planner=PiecewiseLinearPlanner(x0 = x, x_goal =
                                                    x_d, numSteps = 3,
                                                    dt = 0.1))
    (bfe, e), (V, bfv, v), mean, var = cbc2_quadratic_terms(
        lambda u: clf.clc(0, u), x, u0)
    assert to_numpy(bfe @ u0 + e) == pytest.approx(to_numpy(clf.clc(0, u0).mean(x)), abs=1e-4, rel=1e-2)


if __name__ == '__main__':
    test_convert_cbc_terms_to_socp_term()
