import numpy as np
import torch


def controller_sine(xi, m):
    return torch.sin(xi[0]) * torch.abs(torch.rand(m)) + 0.2 * torch.rand(1)


def sample_generator_trajectory(dynamics_model, D, dt=0.01, x0=None,
                                controller=controller_sine):
    f = dynamics_model.f_func
    g = dynamics_model.g_func
    m = dynamics_model.ctrl_size
    n = dynamics_model.state_size
    U = torch.empty((D, m))
    X = torch.zeros((D+1, n))
    X[0, :] = torch.rand(n) if x0 is None else torch.tensor(x0)
    Xdot = torch.zeros((D, n))
    # Single trajectory
    for i in range(D):
        U[i, :] = controller_sine(X[i, :], m)
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

