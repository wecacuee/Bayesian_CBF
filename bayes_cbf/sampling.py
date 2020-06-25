from abc import ABC, abstractmethod

import numpy as np
import torch


def controller_sine(xi, m):
    return torch.sin(xi[0]) * torch.abs(torch.rand(m)) + 0.2 * torch.rand(1)

class Visualizer(ABC):
    @abstractmethod
    def setStateCtrl(self, x, u, t=0):
        pass

class VisualizerZ(Visualizer):
    def setStateCtrl(self, x, u, t=0):
        pass

def sample_generator_trajectory(dynamics_model, D, dt=0.01, x0=None,
                                true_model=None,
                                controller=controller_sine,
                                controller_class=None,
                                visualizer=VisualizerZ()):
    if controller_class is not None:
        controller = controller_class(dt=dt,
                                      true_model=true_model,
                                      plotfile=plotfile.format(suffix='_ctrl_{suffix}'),
                                      dtype=dtype
        ).control


    f = dynamics_model.f_func
    g = dynamics_model.g_func
    m = dynamics_model.ctrl_size
    n = dynamics_model.state_size
    U = torch.empty((D, m))
    X = torch.zeros((D+1, n))
    X[0, :] = torch.rand(n) if x0 is None else (torch.tensor(x0) if not isinstance(x0, torch.Tensor) else x0)
    Xdot = torch.zeros((D, n))
    # Single trajectory
    for i in range(D):
        U[i, :] = controller_sine(X[i, :], m)
        visualizer.setStateCtrl(X[i, :], U[i, :], t=i)
        Xdot[i, :] = f(X[i, :]) + g(X[i, :]) @ U[i, :]
        X[i+1, :] = X[i, :] + Xdot[i, :] * dt
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

