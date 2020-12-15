import os.path as osp
import warnings
from functools import partial

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch
import pytest
import gpytorch.settings as gpsettings

from bayes_cbf.control_affine_model import ControlAffineRegressor, ControlAffineRegressorExact, ControlAffineRegressorVector
from bayes_cbf.plotting import plot_2D_f_func, plot_results, plot_learned_2D_func_from_data
from bayes_cbf.pendulum import PendulumDynamicsModel, plot_learned_2D_func
from bayes_cbf.sampling import sample_generator_independent, sample_generator_trajectory
from bayes_cbf.misc import torch_kron, variable_required_grad, to_numpy, DynamicsModel


PLOTTING = True


class RandomDynamicsModel(DynamicsModel):
    def __init__(self, m, n, deterministic=False, diag_cov=1e-5):
        self.n = n
        self.m = m
        self.deterministic = deterministic
        self.diag_cov = diag_cov
        self.A = torch.rand(n,n)
        self.B = torch.rand(n, m, n)

    @property
    def ctrl_size(self):
        return self.m

    @property
    def state_size(self):
        return self.n

    def f_func(self, X_in):
        A = self.A
        n = self.n
        m = self.m
        if X_in.ndim == 1:
            X = X_in.unsqueeze(0)
        else:
            X = X_in
        deterministic = self.deterministic
        assert X.shape[-1] == n
        cov = torch.eye(n) * self.diag_cov
        mean = A.expand(X.shape[0], *A.shape).bmm(X.unsqueeze(-1)).squeeze(-1)
        fx = (mean if deterministic
              else torch.distributions.MultivariateNormal(mean, cov).sample())
        return fx.squeeze(0) if X_in.ndim == 1 else fx

    def g_func(self, X_in):
        """
        Returns n x m matrix
        """
        B = self.B
        n = self.n
        m = self.m
        if X_in.ndim == 1:
            X = X_in.unsqueeze(0)
        else:
            X = X_in
        deterministic = self.deterministic
        assert X.shape[-1] == n
        cov_A = torch.eye(n) * self.diag_cov
        cov_B = torch.eye(m) * 2 * self.diag_cov
        cov = torch_kron(cov_A.unsqueeze(0), cov_B.unsqueeze(0)).squeeze(0)
        mean = B.reshape(1, -1, self.n).expand(X.shape[0], -1, -1).bmm(X.unsqueeze(-1)).reshape(-1, self.n, self.m)
        gx = (
            mean if deterministic
            else torch.distributions.MultivariateNormal(
                    mean.flatten(), torch.eye(mean.flatten().shape[0]) * self.diag_cov
            ).sample().reshape(-1, n, m)
        )
        return gx.squeeze(0) if X_in.ndim == 1 else gx


def test_GP_train_predict(n=2, m=3,
                          D = 50,
                          deterministic=False,
                          rel_tol=0.10,
                          abs_tol=0.80,
                          perturb_scale=0.1,
                          sample_generator=sample_generator_trajectory,
                          dynamics_model_class=RandomDynamicsModel,
                          training_iter=100,
                          grad_predict=False):
    if grad_predict:
        deterministic = True
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
    shuffled_order = np.arange(D)
    #shuffled_order = torch.randint(D, size=(D,))
    np.random.shuffle(shuffled_order)
    shuffled_order = torch.from_numpy(shuffled_order)
    train_indices = shuffled_order[:int(D*0.8)]
    test_indices = shuffled_order[int(D*0.8):]

    # Train data
    Xtrain, Utrain, XdotTrain = [Mat[train_indices, :]
                                 for Mat in (X, U, Xdot)]
    UHtrain = torch.cat((Utrain.new_ones((Utrain.shape[0], 1)), Utrain), dim=1)
    # Test data
    Xtest, Utest, XdotTest = [Mat[test_indices, :]
                              for Mat in (X, U, Xdot)]

    # Call the training routine
    dgp = ControlAffineRegressor(Xtrain.shape[-1], Utrain.shape[-1])
    dgp_exact = ControlAffineRegressorExact(Xtrain.shape[-1], Utrain.shape[-1])
    dgp_vector = ControlAffineRegressorVector(Xtrain.shape[-1], Utrain.shape[-1])
    # Test prior
    _ = dgp.predict(Xtest, return_cov=False)
    dgp._fit_with_warnings(Xtrain, Utrain, XdotTrain, training_iter=training_iter, lr=0.01)
    _, _ = dgp_exact.custom_predict(Xtest, compute_cov=False)
    dgp_exact._fit_with_warnings(Xtrain, Utrain, XdotTrain, training_iter=training_iter, lr=0.01)
    _, _ = dgp_vector.custom_predict(Xtest, compute_cov=False)
    dgp_vector._fit_with_warnings(Xtrain, Utrain, XdotTrain, training_iter=training_iter, lr=0.01)
    if X.shape[-1] == 2 and U.shape[-1] == 1 and PLOTTING:
        plot_learned_2D_func(Xtrain.detach().cpu().numpy(), dgp.f_func,
                                dynamics_model.f_func)
        plt.savefig('/tmp/f_learned_vs_f_true.pdf')
        plot_learned_2D_func(Xtrain.detach().cpu().numpy(), dgp.g_func,
                                dynamics_model.g_func, axtitle="g(x)[{i}]")
        plt.savefig('/tmp/g_learned_vs_g_true.pdf')

    # check predicting training values
    #FXT_train_mean, FXT_train_cov = dgp.predict(Xtrain)
    #XdotGot_train = XdotTrain.new_empty(XdotTrain.shape)
    #for i in range(Xtrain.shape[0]):
    #    XdotGot_train[i, :] = FXT_train_mean[i, :, :].T @ UHtrain[i, :]
    predict_flatten_deprecated = True
    if not predict_flatten_deprecated:
        XdotGot_train_mean, XdotGot_train_cov = dgp._predict_flatten(
            Xtrain[:-1], Utrain[:-1])
        assert XdotGot_train_mean.detach().cpu().numpy() == pytest.approx(
            XdotTrain[:-1].detach().cpu().numpy(), rel=rel_tol, abs=abs_tol), """
            Train data check using original flatten predict """

    UHtest = torch.cat((Utest.new_ones((Utest.shape[0], 1)), Utest), dim=1)
    if deterministic:
        FXTexpected = torch.empty((Xtest.shape[0], 1+m, n))
        for i in range(Xtest.shape[0]):
            FXTexpected[i, ...] = torch.cat(
                (dynamics_model.f_func(Xtest[i, :])[None, :],
                    dynamics_model.g_func(Xtest[i,  :]).T), dim=0)
            assert torch.allclose(
                XdotTest[i, :], FXTexpected[i, :, :].T @ UHtest[i, :])

    # check predicting train values
    XdotTrain_mean = dgp.fu_func_mean(Utrain[:-1], Xtrain[:-1])
    XdotTrain_mean_exact = dgp_exact.fu_func_mean(Utrain[:-1], Xtrain[:-1])
    XdotTrain_mean_vector = dgp_vector.fu_func_mean(Utrain[:-1], Xtrain[:-1])
    assert XdotTrain_mean.detach().cpu().numpy() == pytest.approx(
        XdotTrain[:-1].detach().cpu().numpy(), rel=rel_tol, abs=abs_tol), """
        Train data check using custom flatten predict """
    assert XdotTrain_mean_exact.detach().cpu().numpy() == pytest.approx(
        XdotTrain[:-1].detach().cpu().numpy(), rel=rel_tol, abs=abs_tol), """
        Train data check using ControlAffineRegressorExact.custom_predict """
    assert XdotTrain_mean_vector.detach().cpu().numpy() == pytest.approx(
        XdotTrain[:-1].detach().cpu().numpy(), rel=rel_tol, abs=abs_tol), """
        Train data check using ControlAffineRegressorExact.custom_predict """

    if grad_predict and n == 1:
        x0 = Xtrain[9:10, :].detach().clone()
        u0 = Utrain[9:10, :].detach().clone()
        #est_grad_fx = dgp.grad_fu_func_mean(x0, u0)
        true_fu_func = lambda X: dynamics_model.f_func(X) + dynamics_model.g_func(X).bmm(u0.unsqueeze(-1)).squeeze(-1)
        with variable_required_grad(x0):
            true_grad_fx = torch.autograd.grad(true_fu_func(x0), x0)[0]
        with variable_required_grad(x0):
            est_grad_fx_2 = torch.autograd.grad(dgp.fu_func_mean(u0, x0), x0)[0]
        assert to_numpy(est_grad_fx_2) == pytest.approx(to_numpy(true_grad_fx), rel=rel_tol, abs=abs_tol)
        #assert to_numpy(est_grad_fx) == pytest.approx(to_numpy(true_grad_fx), rel=rel_tol, abs=abs_tol)

    # Check predicting perturbed train values
    Xptrain = Xtrain[:-1] * (1 + torch.rand(Xtrain.shape[0]-1, 1) * perturb_scale)
    Uptrain = Utrain[:-1] * (1 + torch.rand(Xtrain.shape[0]-1, 1) * perturb_scale)
    Xdot_ptrain = dynamics_model.f_func(Xptrain) + dynamics_model.g_func(Xptrain).bmm(Uptrain.unsqueeze(-1)).squeeze(-1)
    if not predict_flatten_deprecated:
        XdotGot_ptrain_mean, XdotGot_ptrain_cov = dgp._predict_flatten(Xptrain, Uptrain)
        assert XdotGot_ptrain_mean.detach().cpu().numpy() == pytest.approx(
            Xdot_ptrain.detach().cpu().numpy(), rel=rel_tol, abs=abs_tol), """
            Perturbed Train data check using original flatten predict """

    XdotGot_ptrain_mean_custom = dgp.fu_func_mean(Uptrain, Xptrain)
    assert XdotGot_ptrain_mean_custom.detach().cpu().numpy() == pytest.approx(
        Xdot_ptrain.detach().cpu().numpy(), rel=rel_tol, abs=abs_tol), """
        Perturbed Train data check using custom flatten predict """

    # check predicting test values
    # FXTmean, FXTcov = dgp.predict(Xtest)
    # XdotGot = XdotTest.new_empty(XdotTest.shape)
    # for i in range(Xtest.shape[0]):
    #     XdotGot[i, :] = FXTmean[i, :, :].T @ UHtest[i, :]
    if not predict_flatten_deprecated:
        XdotGot_mean, XdotGot_cov = dgp.predict_flatten(Xtest, Utest)
        assert XdotGot_mean.detach().cpu().numpy() == pytest.approx(
            XdotTest.detach().cpu().numpy(), rel=rel_tol, abs=abs_tol)
            #abs=XdotGot_cov.flatten().max())

    # check predicting test values
    Xdot_mean = dgp.fu_func_mean(Utest, Xtest)
    Xdot_mean_exact = dgp_exact.fu_func_mean(Utest, Xtest)
    Xdot_mean_vector = dgp_vector.fu_func_mean(Utest, Xtest)
    assert Xdot_mean.detach().cpu().numpy() == pytest.approx(
        XdotTest.detach().cpu().numpy(), rel=rel_tol, abs=abs_tol)
    assert Xdot_mean_exact.detach().cpu().numpy() == pytest.approx(
        XdotTest.detach().cpu().numpy(), rel=rel_tol, abs=abs_tol)
    assert Xdot_mean_vector.detach().cpu().numpy() == pytest.approx(
        XdotTest.detach().cpu().numpy(), rel=rel_tol, abs=abs_tol)
    return dgp, dynamics_model


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
    dgp.fit(torch.from_numpy(Xtrain[:-1, :]), torch.from_numpy(Utrain),
            torch.from_numpy(XdotTrain))
    dgp.predict(torch.from_numpy(Xtest))


test_GP_train_predict_detrministic = partial(
    test_GP_train_predict,
    deterministic=True,
    sample_generator=sample_generator_independent)
"""
Simplest scenario with random samples generated without a trajectory model and
no randomness.
"""

#@pytest.mark.skip(reason="Not succeding right now. Fix later.")
def test_GP_train_predict_independent():
    """
    Level 2: Simplest scenario with random samples generated without a trajectory model
    """
    test_GP_train_predict(sample_generator=sample_generator_independent)


test_pendulum_train_predict = partial(
    test_GP_train_predict,
    n=2, m=1,
    training_iter=100,
    dynamics_model_class=PendulumDynamicsModel)
"""
Level 4: Pendulum model
"""


test_gp_posterior_derivative = partial(test_GP_train_predict, grad_predict=True, n=1)
"""
Test gradient prediction
"""

def test_exact_matrix_variate_gp():
    pass

if __name__ == '__main__':
    #test_GP_train_predict_detrministic()
    #test_GP_train_predict_independent()
    #test_GP_train_predict()
    #test_control_affine_gp()
    #test_pendulum_train_predict()
    test_gp_posterior_derivative()

