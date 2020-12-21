from abc import ABC, abstractmethod

import numpy as np
import torch


def controller_sine(xi, t=1):
    m = 1
    return torch.sin(xi[0]) * torch.abs(torch.rand(m)) + 0.2 * torch.rand(1)

class Visualizer(ABC):
    @abstractmethod
    def setStateCtrl(self, x, u, t=0, **kw):
        pass

class VisualizerZ(Visualizer):
    def setStateCtrl(self, x, u, t=0, **kw):
        pass

def uncertainity_vis_kwargs(controller, x, u, dt):
    if (hasattr(controller, "__self__")
        and hasattr(controller.__self__, "model")
        and hasattr(controller.__self__.model, "fu_func_gp")):
        xtp1_gp = controller.__self__.model.fu_func_gp(u)
        xtp1 = xtp1_gp.mean(x) * dt + x
        xtp1_var = xtp1_gp.knl(x, x) * dt * dt
        vis_kwargs = dict(xtp1=xtp1, xtp1_var=xtp1_var)
    else:
        vis_kwargs = dict()
    return vis_kwargs

def sample_generator_trajectory(dynamics_model, D, dt=0.01, x0=None,
                                true_model=None,
                                controller=controller_sine,
                                controller_class=None,
                                visualizer=VisualizerZ()):
    if controller_class is not None:
        controller = controller_class(dt=dt,
                                      true_model=true_model
        ).control


    m = dynamics_model.ctrl_size
    n = dynamics_model.state_size
    U = torch.empty((D, m))
    X = torch.zeros((D+1, n))
    X[0, :] = torch.rand(n) if x0 is None else (torch.tensor(x0) if not isinstance(x0, torch.Tensor) else x0)
    Xdot = torch.zeros((D, n))
    # Single trajectory
    dynamics_model.set_init_state(X[0, :])
    for t in range(D):
        U[t, :] = controller(X[t, :], t=t)
        visualizer.setStateCtrl(X[t, :], U[t, :], t=t,
                                **uncertainity_vis_kwargs(controller, X[t, :], U[t, :], dt))
        obs = dynamics_model.step(U[t, :], dt)
        Xdot[t, :] = obs['xdot'] # f(X[t, :]) + g(X[t, :]) @ U[t, :]
        X[t+1, :] = obs['x'] # normalize_state(X[t, :] + Xdot[t, :] * dt)
    return Xdot, X, U


def sample_generator_independent(dynamics_model, D):
    # Idependent random mappings
    f = dynamics_model.f_func
    g = dynamics_model.g_func
    m = dynamics_model.ctrl_size
    n = dynamics_model.state_size

    U = torch.rand(D, m)
    X = torch.rand(D, n)
    Xdot = torch.zeros((D, n))
    for i in range(D):
        Xdot[i, :] = f(X[i, :]) + g(X[i, :]) @ U[i, :]
    return Xdot, X, U

