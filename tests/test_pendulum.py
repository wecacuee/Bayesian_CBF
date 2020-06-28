import torch
import numpy as np
from bayes_cbf.pendulum import PendulumDynamicsModel, EnergyCLF, RadialCBFRelDegree2


def test_EnergyCLF_grad_V_clf():
    model = PendulumDynamicsModel(m=1, n=2)
    eclf = EnergyCLF(model)
    x = torch.rand(2)
    eclf.grad_V_clf(x)


def test_RadialCBF_grad_h_col():
    model = PendulumDynamicsModel(m=1, n=2)
    rcol = RadialCBFRelDegree2(model)
    xt = torch.rand(2)
    with torch.no_grad():
        grad_h_x = rcol.grad_cbf(xt)
    xt.requires_grad_(True)
    torch_grad_h_x = torch.autograd.grad(rcol.cbf(xt), xt)[0]
    assert torch.allclose(grad_h_x, torch_grad_h_x)
