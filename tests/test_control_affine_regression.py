import os.path as osp
import warnings
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import torch
import pytest
import gpytorch.settings as gpsettings

from bayes_cbf.control_affine_model import ControlAffineRegressor


def rad2deg(rad):
    return rad / np.pi * 180


def plot_results(time_vec, omega_vec, theta_vec, u_vec):
    #plot thetha
    fig, axs = plt.subplots(2,2)
    axs[0,0].plot(time_vec, rad2deg(theta_vec),
                  ":", label = "theta (degrees)",color="blue")
    axs[0,0].set_ylabel("theta (degrees)")
    axs[0,1].plot(time_vec, omega_vec,":", label = "omega (rad/s)",color="blue")
    axs[0,1].set_ylabel("omega")
    axs[1,0].plot(time_vec, u_vec,":", label = "u",color="blue")
    axs[1,0].set_ylabel("u")

    axs[1,1].plot(time_vec, np.cos(theta_vec),":", label="cos(theta)", color="blue")
    axs[1,1].set_ylabel("cos/sin(theta)")
    axs[1,1].plot(time_vec, np.sin(theta_vec),":", label="sin(theta)", color="red")
    axs[1,1].set_ylabel("sin(theta)")
    axs[1,1].legend()

    fig.suptitle("Pendulum")
    fig.subplots_adjust(wspace=0.31)
    plt.show()


def sample_generator_trajectory(f, g, D, n, m, dt=0.001):
    U = np.random.rand(D, m)
    X = np.zeros((D+1, n))
    X[0, :] = np.random.rand(n)
    Xdot = np.zeros((D, n))
    # Single trajectory
    for i in range(D):
        Xdot[i, :] = f(X[i, :]) + g(X[i, :]) @ U[i, :]
        X[i+1, :] = X[i, :] + Xdot[i, :] * dt
    return Xdot, X, U


def sample_generator_independent(f, g, D, n, m):
    # Idependent random mappings
    U = np.random.rand(D, m)
    X = np.random.rand(D, n)
    Xdot = np.zeros((D, n))
    for i in range(D):
        Xdot[i, :] = f(X[i, :]) + g(X[i, :]) @ U[i, :]
    return Xdot, X, U


class RandomDynamicsModel:
    def __init__(self, m, n, deterministic=False):
        self.n = n
        self.m = m
        self.deterministic = deterministic
        self.A = np.random.rand(n,n)
        self.B = np.random.rand(n, m, n)

    def f(self, x):
        A = self.A
        n = self.n
        m = self.m
        deterministic = self.deterministic
        assert x.shape[-1] == n
        cov = np.eye(n) * 0.0001
        return (A @ x if deterministic
                else np.random.multivariate_normal(A @ x, cov))

    def g(self, x):
        """
        Returns n x m matrix
        """
        B = self.B
        n = self.n
        m = self.m
        deterministic = self.deterministic
        assert x.shape[-1] == n
        cov_A = np.eye(n) * 0.0001
        cov_B = np.eye(m) * 0.0002
        cov = np.kron(cov_A, cov_B)

        return (
            B @ x if deterministic
            else np.random.multivariate_normal(
                    (B @ x).flatten(), cov
            ).reshape((n, m))
        )


class PendulumDynamicsModel:
    def __init__(self, m, n, mass=1, gravity=10, length=1, deterministic=True):
        self.m = m
        self.n = n
        self.mass = mass
        self.gravity = gravity
        self.length = length

    def f(self, X):
        m = self.m
        n = self.n
        mass = self.mass
        gravity = self.gravity
        length = self.length
        X = np.asarray(X)
        theta_old, omega_old = X[..., 0:1], X[..., 1:2]
        return np.concatenate([omega_old,
                               - (gravity/length)*np.sin(theta_old)], axis=-1)

    def g(self, x):
        m = self.m
        n = self.n
        mass = self.mass
        gravity = self.gravity
        length = self.length
        return np.array([[0], [1/(mass*length)]])


def test_GP_train_predict(n=2, m=3,
                          deterministic=False,
                          rel_tol=0.05,
                          abs_tol=0.05,
                          sample_generator=sample_generator_trajectory,
                          dynamics_model_class=RandomDynamicsModel):
    #chosen_seed = np.random.randint(100000)
    chosen_seed = 52648
    print(chosen_seed)
    np.random.seed(chosen_seed)
    torch.manual_seed(chosen_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Collect training data
    D = 20
    dynamics_model = dynamics_model_class(m, n, deterministic=deterministic)
    Xdot, X, U = sample_generator(dynamics_model.f,
                                  dynamics_model.g,
                                  D, n, m)
    if X.shape[-1] == 2 and U.shape[-1] == 1:
        plot_results(np.arange(U.shape[0]),
                     omega_vec=X[:-1, 0],
                     theta_vec=X[:-1, 1],
                     u_vec=U[:, 0])

    # Test train split
    shuffled_order = np.arange(D)
    np.random.shuffle(shuffled_order)
    train_indices = shuffled_order[:int(D*0.8)]
    test_indices = shuffled_order[int(D*0.8):]

    # Train data
    Xtrain, Utrain, XdotTrain = [Mat[train_indices, :]
                                 for Mat in (X, U, Xdot)]
    # Test data
    Xtest, Utest, XdotTest = [Mat[test_indices, :]
                              for Mat in (X, U, Xdot)]

    # Call the training routine
    dgp = ControlAffineRegressor(Xtrain.shape[-1], Utrain.shape[-1])
    # Test prior
    _ = dgp.predict(Xtest, return_cov=False)
    dgp.fit(Xtrain, Utrain, XdotTrain, training_iter=50, lr=0.01)

    UHtest = np.concatenate((np.ones((Utest.shape[0], 1)), Utest), axis=1)
    if deterministic:
        FXTexpected = np.empty((Xtest.shape[0], 1+m, n))
        for i in range(Xtest.shape[0]):
            FXTexpected[i, ...] = np.concatenate(
                (f(Xtest[i, :])[None, :], g(Xtest[i,  :]).T), axis=0)
            assert np.allclose(XdotTest[i, :], FXTexpected[i, :, :].T @ UHtest[i, :])

    #FXTexpected = np.concatenate(((f.A @ Xtest.T).T.reshape(-1, 1, n),
    #                              (g.B @ Xtest.T).T.reshape(-1, m, n)), axis=1)
    FXTmean, FXTcov = dgp.predict(Xtest)

    XdotGot = np.empty_like(XdotTest)
    for i in range(Xtest.shape[0]):
        XdotGot[i, :] = FXTmean[i, :, :].T @ UHtest[i, :]
    assert XdotGot == pytest.approx(XdotTest, rel=rel_tol, abs=abs_tol)


def relpath(path,
            root=osp.dirname(__file__) or '.'):
    return osp.join(root, path)


def test_control_affine_gp(
        datasrc=relpath('data/Xtrain_Utrain_X_interpolate_lazy_tensor_error.npz')):
    loaded_data = np.load(datasrc)
    Xtrain = loaded_data['Xtrain']
    Utrain = loaded_data['Utrain']
    Xtest = loaded_data['X']
    XdotTrain = Xtrain[1:, :] - Xtrain[:-1, :]
    dgp = ControlAffineRegressor(Xtrain.shape[-1], Utrain.shape[-1])
    dgp.fit(Xtrain[:-1, :], Utrain, XdotTrain)
    dgp.predict(Xtest)


test_pendulum_train_predict = partial(
    test_GP_train_predict,
    n=2, m=1,
    dynamics_model_class=PendulumDynamicsModel)


test_pendulum_train_predict_trajectory = partial(
    test_pendulum_train_predict,
    sample_generator=sample_generator_independent)


if __name__ == '__main__':
    #test_GP_train_predict()
    #test_control_affine_gp()
    test_pendulum_train_predict()

