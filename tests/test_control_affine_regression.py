import os.path as osp
import warnings
from functools import partial

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch
import pytest
import gpytorch.settings as gpsettings

from bayes_cbf.control_affine_model import ControlAffineRegressor
from bayes_cbf.plotting import plot_2D_f_func, plot_results, plot_learned_2D_func
from bayes_cbf.pendulum import PendulumDynamicsModel
from bayes_cbf.sampling import sample_generator_independent, sample_generator_trajectory


class RandomDynamicsModel:
    def __init__(self, m, n, deterministic=False):
        self.n = n
        self.m = m
        self.deterministic = deterministic
        self.A = torch.rand(n,n)
        self.B = torch.rand(n, m, n)

    @property
    def ctrl_size(self):
        return self.m

    @property
    def state_size(self):
        return self.n

    def f_func(self, x):
        A = self.A
        n = self.n
        m = self.m
        deterministic = self.deterministic
        assert x.shape[-1] == n
        cov = torch.eye(n) * 0.0001
        return (A @ x if deterministic
                else torch.distributions.MultivariateNormal(A @ x, cov).sample())

    def g_func(self, x):
        """
        Returns n x m matrix
        """
        B = self.B
        n = self.n
        m = self.m
        deterministic = self.deterministic
        assert x.shape[-1] == n
        cov_A = torch.eye(n) * 0.0001
        cov_B = torch.eye(m) * 0.0002
        cov = (cov_A.reshape(n, 1, n, 1) * cov_B.reshape(1, m, 1, m)).reshape(n*m, n*m)

        return (
            B @ x if deterministic
            else torch.distributions.MultivariateNormal(
                    (B @ x).flatten(), cov
            ).sample().reshape((n, m))
        )


def test_GP_train_predict(n=2, m=3,
                          D = 20,
                          deterministic=False,
                          rel_tol=0.10,
                          abs_tol=0.10,
                          sample_generator=sample_generator_trajectory,
                          dynamics_model_class=RandomDynamicsModel):
    chosen_seed = torch.randint(100000, (1,))
    #chosen_seed = 52648
    print("Random seed: {}".format(chosen_seed))
    torch.manual_seed(chosen_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Collect training data
    dynamics_model = dynamics_model_class(m, n, deterministic=deterministic)
    Xdot, X, U = sample_generator(dynamics_model, D)
    if X.shape[-1] == 2 and U.shape[-1] == 1:
        plot_results(torch.arange(U.shape[0]),
                     omega_vec=X[:-1, 0],
                     theta_vec=X[:-1, 1],
                     u_vec=U[:, 0])

    # Test train split
    shuffled_order = torch.arange(D)
    #torch.random.shuffle(shuffled_order)
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
    with torch.no_grad():
        if X.shape[-1] == 2 and U.shape[-1] == 1:
            plot_learned_2D_func(Xtrain.detach().cpu().numpy(), dgp.f_func, dynamics_model.f_func)
            plt.savefig('f_learned_vs_f_true.pdf')
            plot_learned_2D_func(Xtrain.detach().cpu().numpy(), dgp.g_func, dynamics_model.g_func, axtitle="g(x)[{i}]")
            plt.savefig('g_learned_vs_g_true.pdf')

        UHtest = torch.cat((torch.ones((Utest.shape[0], 1)), Utest), axis=1)
        if deterministic:
            FXTexpected = torch.empty((Xtest.shape[0], 1+m, n))
            for i in range(Xtest.shape[0]):
                FXTexpected[i, ...] = torch.cat(
                    (f(Xtest[i, :])[None, :], g(Xtest[i,  :]).T), axis=0)
                assert torch.allclose(XdotTest[i, :], FXTexpected[i, :, :].T @ UHtest[i, :])

        #FXTexpected = np.concatenate(((f.A @ Xtest.T).T.reshape(-1, 1, n),
        #                              (g.B @ Xtest.T).T.reshape(-1, m, n)), axis=1)
        FXTmean, FXTcov = dgp.predict(Xtest)

        XdotGot = XdotTest.new_empty(XdotTest.shape)
        for i in range(Xtest.shape[0]):
            XdotGot[i, :] = FXTmean[i, :, :].T @ UHtest[i, :]
        assert XdotGot.detach().cpu().numpy() == pytest.approx(XdotTest.detach().cpu().numpy(), rel=rel_tol, abs=abs_tol)


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
    D=200,
    dynamics_model_class=PendulumDynamicsModel)


if __name__ == '__main__':
    #test_GP_train_predict()
    #test_control_affine_gp()
    test_pendulum_train_predict()

