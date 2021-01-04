import pytest
import torch
from bayes_cbf.controllers import SOCPController, SumDynamicModels
from bayes_cbf.planner import PiecewiseLinearPlanner
from bayes_cbf.misc import to_numpy, random_psd
from bayes_cbf.unicycle_move_to_pose import (ControllerCLFBayesian,
                                             PiecewiseLinearPlanner,
                                             LearnedShiftInvariantDynamics,
                                             CLFCartesian,
                                             obstacles_at_mid_from_start_and_goal,
                                             AckermanDrive)
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
    x_g = torch.rand(n)
    u0 = torch.rand(m)
    x = torch.rand(n)
    dt = 0.01
    cbf_gammas = [5., 5.]
    Kp = [0.9, 1.5, 0.]
    numSteps = 200
    clf = CLFCartesian(Kp = torch.tensor(Kp))
    ctrller = ControllerCLFBayesian(
            PiecewiseLinearPlanner(x, x_g, numSteps, dt),
            coordinate_converter = lambda x, x_g: x,
            dynamics = LearnedShiftInvariantDynamics(dt = dt,
                                                     mean_dynamics = AckermanDrive()),
            clf = clf,
            cbfs = obstacles_at_mid_from_start_and_goal(x , x_g),
            cbf_gammas = torch.tensor(cbf_gammas)

    )
    state = x
    state_goal = x_g
    t = 20
    (bfe, e), (V, bfv, v), mean, var = cbc2_quadratic_terms(
        lambda u: ctrller._clc(state, state_goal, u, t) * -1.0,
        state, torch.rand(m))
    assert to_numpy(bfe @ u0 + e) == pytest.approx(to_numpy(ctrller._clc(x, x_g, u0, t)), abs=1e-4, rel=1e-2)


if __name__ == '__main__':
    test_convert_cbc_terms_to_socp_term()
