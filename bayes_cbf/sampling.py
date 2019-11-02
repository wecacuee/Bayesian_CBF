import numpy as np


def controller_sine(xi, m):
    return np.sin(xi[0]) * np.abs(np.random.rand(m)) + 0.2 * np.random.rand()


def sample_generator_trajectory(dynamics_model, D, dt=0.01, x0=None,
                                controller=controller_sine):
    f = dynamics_model.f_func
    g = dynamics_model.g_func
    m = dynamics_model.ctrl_size
    n = dynamics_model.state_size
    U = np.empty((D, m))
    X = np.zeros((D+1, n))
    X[0, :] = np.random.rand(n) if x0 is None else np.asarray(x0)
    Xdot = np.zeros((D, n))
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

    U = np.random.rand(D, m)
    X = np.random.rand(D, n)
    Xdot = np.zeros((D, n))
    for i in range(D):
        Xdot[i, :] = f(X[i, :]) + g(X[i, :]) @ U[i, :]
    return Xdot, X, U

