import torch
import numpy as np
from bayes_cbf.pendulum import PendulumDynamicsModel, EnergyCLF, RadialCBF


def test_EnergyCLF_grad_V_clf():
    model = PendulumDynamicsModel(m=1, n=2)
    eclf = EnergyCLF(model)
    x = np.random.rand(2)
    eclf.grad_V_clf(x)


def test_RadialCBF_grad_h_col():
    model = PendulumDynamicsModel(m=1, n=2)
    rcol = RadialCBF(model)
    x = np.random.rand(2)
    xt = torch.from_numpy(x).requires_grad_(True)
    for key, val in vars(rcol).items():
        if isinstance(val, float) or isinstance(val, np.ndarray):
            setattr(rcol, key, torch.tensor(val))
    rcol.h_col(xt, pkg=torch).backward()
    assert np.allclose(rcol.grad_h_col(x), xt.grad)
